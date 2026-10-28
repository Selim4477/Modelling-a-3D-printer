import numpy as np
import matplotlib.pyplot as plt
from Parameters import *
from Model import x_c, y_c, a_e, b_e, phi, T_end

# Parameter (Filament)
rho_f        = 1250        # Filament density [kg/m^3]
c_f          = 1800        # Filament specific heat capacity [J/(kg*K)]
k_f          = 0.13        # Filament thermal conductivity [W/(m*K)]
sl_f         = 1.75e-3     # Side length of filament [m]
h_b          = 80          # Heat transfer coefficient to bed [W/(K*m^2)]
h_e          = 12          # Heat transfer coefficient to environment [W/(K*m^2)]
uni_tol      = 0.1         # Uniform temperature tolerance [deg C]
delta_t      = 0.02        # Discrete time step [s]

def filament_pde(L_total, T_nozzle_profile, dt = 0.02, N = 200):
    alpha = k_f / (rho_f * c_f)
    coeff = (h_b + 3*h_e) / (sl_f * rho_f * c_f)
    const = (h_b * T_b + 3 * h_e * T_e) / (sl_f * rho_f * c_f)
    ds = L_total / N
    t_print = L_total / v_p

    # F3 Fix: Initialize unprinted nodes as NaN instead of T_e
    T = np.ones(N) * np.nan
    T_hist = [T.copy()]
    times = [0.0]
    active_l = [0.0]

    t=0.0
    n_prev = 0
    while t < t_print:
        t_next = min(t + dt, t_print)
        L_active = min(v_p*t_next, L_total)
        n_active = max(2, int(np.ceil(L_active/ds)))
        n_active = min(n_active, N)
        T_dep = np.interp(t_next, T_nozzle_profile[0], T_nozzle_profile[1])
        
        # If nodes were previously NaN, initialize them to T_dep or T_e before PDE stepping
        for idx in range(n_prev, n_active):
            if np.isnan(T[idx]):
                T[idx] = T_e
        
        T[n_prev:n_active] = T_dep
        n_prev = n_active

        T_new = T.copy()
        
        # Only compute PDE for active nodes that are not NaN
        active_indices = np.where(~np.isnan(T[:n_active]))[0]
        if len(active_indices) > 0:
            Ta = T[:n_active].copy()
            if n_active == 2:
                T_new[0] = Ta[0]
                T_new[1] = T_dep
            else:
                d2T0 = 2*(Ta[1]-Ta[0]) / ds**2
                dT_dt0 = alpha * d2T0 - coeff * Ta[0] + const
                T_new[0] = Ta[0] + dt * dT_dt0

                for i in range(1, n_active - 1):
                    d2T = (Ta[i+1] - 2*Ta[i] + Ta[i-1]) / ds**2
                    dT_dt = alpha * d2T - coeff * Ta[i] + const
                    T_new[i] = Ta[i] + dt * dT_dt

                T_new[n_active - 1] = T_dep

        # F3 Fix: Keep unprinted future nodes as NaN
        if n_active < N:
            T_new[n_active:] = np.nan

        T = T_new
        t = t_next
        T_hist.append(T.copy())
        times.append(t)
        active_l.append(L_active)

    while True:
        T_new = T.copy()
        for i in range(N):
            im = (i - 1) % N
            ip = (i + 1) % N
            d2T = (T[ip] - 2*T[i] + T[im]) / ds**2
            dT_dt = alpha * d2T - coeff * T[i] + const
            T_new[i] = T[i] + dt * dT_dt

        T = T_new
        t += dt
        T_hist.append(T.copy())
        times.append(t)
        active_l.append(L_total)

        if np.max(np.nan_to_num(T)) - np.min(np.nan_to_num(T)) < uni_tol or t > t_print + 3000:
            break

    return {"times": np.array(times), "T_hist": np.array(T_hist), "active_l": np.array(active_l), "s_grid": np.linspace(0, L_total, N, endpoint = False), "t_print": t_print, "t_end": t}

X = x_c + a_e*np.cos(phi)
Y = y_c + b_e*np.sin(phi)

dX = np.diff(X)
dY = np.diff(Y)
d_length = np.sqrt(dX**2 + dY**2)
perimeter = np.sum(d_length)
v_p = perimeter/T_end

pde = filament_pde(perimeter, (np.array([0.0, T_end]), np.array([T_n_ref, T_n_ref])), dt = delta_t, N = 200)
print(f"Printing time: {pde['t_print']:.2f}s, filament nearly uniform after {pde['t_end']:.1f}s ({pde['t_end'] - pde['t_print']:.1f}s after printing).")

t_pde = pde["times"]
T_hist = pde["T_hist"]
s_mm = pde["s_grid"] * 1e3

# --- Figure 1: Filament Cooling along Arc Length ---
plt.figure(figsize=(10, 5), constrained_layout=True)

# F4 Fix: Choose explicit times including the printing phase (e.g., t = 0s, 1s, 2s, 43.6s, end)
target_times = [0.0, 1.0, 2.0, 4.0, pde['t_print']/2, pde['t_print'], pde['t_end']]
idxs = [np.abs(t_pde - t_target).argmin() for t_target in target_times]
# Remove duplicate indices if any time points map to the same discrete step
idxs = sorted(list(set(idxs)))

for k in idxs:
    plt.plot(s_mm, T_hist[k], label=f't={t_pde[k]:.1f}s')
plt.title("Filament Cooling along Arc Length")
plt.xlabel("Arc Length [mm]")
plt.ylabel("Temperature [°C]")
plt.legend(fontsize=8)
plt.grid(True)

# --- Figure 2: Thermal Map ---
plt.figure(figsize=(10, 5), constrained_layout=True)
im = plt.imshow(T_hist.T, aspect='auto', origin='lower', extent=[pde["times"][0], pde["times"][-1], s_mm[0], s_mm[-1]], cmap='plasma')
plt.title("1D Distributed Filament Cooling Thermal Map")
plt.xlabel("Time [s]")
plt.ylabel("Arc Length [mm]")
plt.grid(False)
cbar = plt.colorbar(im)
cbar.set_label("Temperature [°C]")

# --- Figure 3: Thermal Map (First 10 Seconds) ---
plt.figure(figsize=(10, 5), constrained_layout=True)
mask_10s = t_pde <= 10.0
t_sub = t_pde[mask_10s]
T_hist_sub = T_hist[mask_10s]

im3 = plt.imshow(T_hist_sub.T, aspect='auto', origin='lower', 
                 extent=[t_sub[0], t_sub[-1], s_mm[0], s_mm[-1]], cmap='plasma')
plt.title("1D Distributed Filament Cooling Thermal Map (First 10s)")
plt.xlabel("Time [s]")
plt.ylabel("Arc Length [mm]")
plt.grid(False)
cbar3 = plt.colorbar(im3)
cbar3.set_label("Temperature [°C]")

plt.show()