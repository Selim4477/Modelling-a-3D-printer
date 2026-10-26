#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Parameters import *

# To simplify the SS, new parameters were defined
a = J_m + (m *r_m**2)/2 +(m_b*r_m**2)/4
c = -(m_b*r_m**2)/4
delta = a**2 -c**2
b1 = ((r_m**2)/4) * (mu_lx + mu_ly + 12*mu_r)
b2 = ((r_m**2)/4) * (mu_lx - mu_ly + 4*mu_r)
p1 = (a*b1 - c*b2)/delta
p2 = (a*b2 - c*b1)/delta
k1 = a*K_m/delta
k2 = -c*K_m/delta

# Angle to position conversion
def conv(theta_l, theta_r):
    x = r_m*(theta_l+theta_r)/2
    y = r_m*(theta_l-theta_r)/2
    return x,y

# The state vector
#x_vec = np.array([theta_l, theta_r, omega_l, omega_r, i_l, i_r]).reshape(-1, 1)

# ------------------------- Constructing the SS model ------------------------
# A matrix
A = np.array([
    [0, 0, 1,         0,         0,          0],
    [0, 0, 0,         1,         0,          0],
    [0, 0, -p1,       -p2,       k1,         k2],
    [0, 0, -p2,       -p1,       k2,         k1],
    [0, 0, -K_m/L_m,  0,         -R_m/L_m,   0],
    [0, 0, 0,         -K_m/L_m,  0,          -R_m/L_m],
])

# B matrix
B = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1/L_m, 0],
    [0, 1/L_m],
])

# Time generation
t_span = (0.0, 1.0)

# Initial condition
x_0 = np.zeros(6)

# Input vector u
u = np.array([[5.0],
              [-2.0]])

def SS(A,B,t_span,x_0,u):
    def state_derivative(t, x):
        x = x.reshape(-1, 1)
        x_dot = A @ x + B @ u
        return x_dot.flatten()
    solution = solve_ivp(state_derivative, t_span, x_0, t_eval=np.linspace(t_span[0], t_span[1], 1000))
    return solution

sol = SS(A, B, t_span, x_0, u)
time_array = sol.t
theta_l = sol.y[0, :]
theta_r = sol.y[1, :]

pos_x, pos_y = conv(theta_l, theta_r)

# ------------------------ Decoupled system Q1.3 -----------------------------
M_x = m + 2*J_m/r_m**2
M_y = m + m_b + 2*J_m/r_m**2
b_x = mu_lx + 8*mu_r
b_y = mu_ly + 4*mu_r

# Decoupled x
A_x = np.array([
    [0, 1,                  0],
    [0, -b_x/M_x,           2*K_m/(r_m*M_x)],
    [0, -K_m/(r_m*L_m),     -R_m/L_m],
])
B_x = np.array([[0], [0], [1/L_m]])

# Decoupled y
A_y = np.array([
    [0, 1, 0],
    [0, -b_y/M_y, 2*K_m/(r_m*M_y)],
    [0, -K_m/(r_m*L_m), -R_m/L_m],
])
B_y = np.array([[0], [0], [1/L_m]])

# -------------------- Q1.4 --------------------
# Defining phi and time sequence
def timing(T_end):
    t = np.linspace(0, T_end, 20001)
    w = 2*np.pi/T_end
    phi = w*t - np.sin(w*t)
    return t, phi

# First, second and third derivative
def derivatives(d, t):
    d1 = np.gradient(d, t)
    d2 = np.gradient(d1, t)
    d3 = np.gradient(d2, t)
    return d1, d2, d3

# Solving the decoupled equations of one axis backwards
def inversion(vel, acc, jerk, M, b):
    i = r_m * (M * acc + b * vel) / (2 * K_m)
    di_dt = r_m * (M * jerk + b * acc) / (2 * K_m)
    u_axis = L_m * di_dt + R_m * i + (K_m / r_m) * vel
    return u_axis, i

# Finding the corresponding voltages
def motor_voltages(X, Y, t):
    Xd, Xdd, Xddd = derivatives(X, t)
    Yd, Ydd, Yddd = derivatives(Y, t)
    u_x, i_x = inversion(Xd, Xdd, Xddd, M_x, b_x)
    u_y, i_y = inversion(Yd, Ydd, Yddd, M_y, b_y)
    u_l = u_x + u_y
    u_r = u_x - u_y
    return u_l, u_r

# Converting x and y back to theta
def conv_inv(x, y):
    theta_l = (x + y)/r_m
    theta_r = (x - y)/r_m
    return theta_l, theta_r

# Defining the simulation with the new input
def simulation(X, Y, t, u_l, u_r):
    def state_derivative(time, x):
        u_n = np.array([np.interp(time, t, u_l), np.interp(time, t, u_r)])
        return A @ x + B @ u_n

    theta_l0, theta_r0 = conv_inv(X[0], Y[0])
    x_start = np.array([theta_l0, theta_r0, 0, 0, 0, 0]) # Starting from rest

    t_out = np.linspace(t[0], t[-1], 1000)
    solution = solve_ivp(state_derivative, (t[0], t[-1]), x_start, t_eval=t_out,
                         rtol=1e-8, atol=1e-10)
    x_sim, y_sim = conv(solution.y[0], solution.y[1])
    return x_sim, y_sim

# Plotting the result (AI assistance)
plt.rcParams["axes.grid"] = True
BLUE, ORANGE = "#2a78d6", "#eb6834"  # left motor / x  and  right motor / y
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"
def plot_shape(name, X, Y, t, u_l, u_r, x_sim, y_sim):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    fig.suptitle(name, fontsize=13, fontweight="bold", x=0.01, ha="left")

    # (a) wanted and simulated path
    ax1 = ax[0]
    ax1.add_patch(plt.Rectangle((0, 0), d_x*1e3, d_y*1e3, facecolor="#f2f2f2",
                                edgecolor="#9a9a9a", linewidth=1.2, zorder=0,
                                label="print bed"))
    ax1.plot(X*1e3, Y*1e3, color=MUTED, linestyle="--", linewidth=1.5, label="wanted path")
    ax1.plot(x_sim*1e3, y_sim*1e3, color=BLUE, alpha=0.85, label="simulated path")
    ax1.plot(X[0]*1e3, Y[0]*1e3, "o", color=INK, markersize=7, label="start / end")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title("Path of the head")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.legend(loc="upper right", fontsize=8.5)

    # (b) motor voltages
    ax2 = ax[1]
    ax2.plot(t, u_l, color=BLUE, label="$u_l$ (left)")
    ax2.plot(t, u_r, color=ORANGE, label="$u_r$ (right)")
    ax2.axhline(0, color="#b8b8b8", linewidth=1)
    ax2.set_title("Motor voltages needed")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("voltage [V]")
    ax2.legend(loc="best", fontsize=8.5)
# ----------------- Defining the ellipse ------------------
T_end = 4.0
t, phi = timing(T_end)

x_c = 0.125
y_c = 0.125
a_e = 0.030
b_e = 0.020
if __name__ == "__main__":
    X = x_c + a_e*np.cos(phi)
    Y = y_c + b_e*np.sin(phi)

    u_l, u_r = motor_voltages(X, Y, t)
    x_sim, y_sim = simulation(X, Y, t, u_l, u_r)

    plot_shape("Ellipse", X, Y, t, u_l, u_r, x_sim, y_sim)
    plt.show()

# ------------------ Infinity symbol -------------------
A_8 = 0.070
B_8 = 0.040
if __name__ == "__main__":
    X_8 = x_c + A_8*np.sin(phi)
    Y_8 = y_c + B_8*np.sin(2*phi)

    u_l_8, u_r_8 = motor_voltages(X_8, Y_8, t)
    x_8, y_8 = simulation(X_8, Y_8, t, u_l_8, u_r_8)

    plot_shape("Infinity", X_8, Y_8, t, u_l_8, u_r_8, x_8, y_8)
    plt.show()

# ----------------- Q 1.5 --------------------
# Construction of state derivative of the nozzle dynamics
def nozzle(t, x_n, u_n):
    T_n = x_n[0]
    i_n = x_n[1]
    v_C = x_n[2]

    dT_n = (v_C**2 / R_n - (T_n - T_e) / R_tn) / C_tn
    di_n = (u_n - v_C) / L_n
    dv_C = (i_n - v_C / R_n) / C_n
    return np.array([dT_n, di_n, dv_C])

# Voltage needed to keep the temperature constant
u_hold = np.sqrt(R_n * (T_n_ref - T_e) / R_tn)
# Initial state of the nozzle
if __name__ == "__main__":
    x_n0 = np.array([T_e, 0, 0])

    # 2 different time scales dynamics occur, the electrical and thermal parts
    # Electrical dynamics
    t_a = np.linspace(0, 3e-3, 1000)
    sol_a = solve_ivp(nozzle, (0, 3e-3), x_n0, args=(u_n_max,), t_eval=t_a,
                      method="Radau", rtol=1e-8, atol=1e-10)
    # Thermal dynamics
    # If the temperature is below the reference, use full voltage to get there, if reached, keep the voltage the same
    def heating(t, x_n):
        T_n = x_n[0]

        if T_n < T_n_ref:
            u_n = u_n_max
        else:
            u_n = u_hold

        return nozzle(t, x_n, u_n)
    # Solve the ODE
    t_h = np.linspace(0, 300, 601)
    sol_n = solve_ivp(heating, (0,300), x_n0, t_eval=t_h,
                          method="Radau", rtol=1e-6, atol=1e-8)
    T_sim = sol_n.y[0]
    u_b = np.where(T_sim < T_n_ref, u_n_max, u_hold)


    # ------------------ Plotting (AI assistance) --------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig.suptitle("Nozzle heater", fontsize=13, fontweight="bold", x=0.01, ha="left")

    # (a) electrical start-up: heater voltage
    ax1 = ax[0, 0]
    ax1.plot(t_a * 1e3, sol_a.y[2], color=BLUE, label="$v_C$")
    ax1.axhline(u_n_max, color=MUTED, linestyle="--", linewidth=1, label="supply voltage")
    ax1.set_title("Start-up: heater voltage $v_C$")
    ax1.set_xlabel("time [ms]")
    ax1.set_ylabel("voltage [V]")
    ax1.legend(loc="lower right", fontsize=8.5)

    # (b) electrical start-up: current
    ax2 = ax[0, 1]
    ax2.plot(t_a * 1e3, sol_a.y[1], color=ORANGE, label="$i_n$")
    ax2.axhline(u_n_max / R_n, color=MUTED, linestyle="--", linewidth=1, label="steady state $u_n/R_n$")
    ax2.set_title("Start-up: current $i_n$")
    ax2.set_xlabel("time [ms]")
    ax2.set_ylabel("current [A]")
    ax2.legend(loc="lower right", fontsize=8.5)

    # (c) temperature
    ax3 = ax[1, 0]
    ax3.plot(t_h, T_sim, color=ORANGE, label="$T_n$")
    ax3.axhline(T_n_ref, color=MUTED, linestyle="--", linewidth=1, label="target")
    ax3.set_title("Heating up: nozzle temperature")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("temperature [C]")
    ax3.legend(loc="lower right", fontsize=8.5)

    # (d) applied voltage
    ax4 = ax[1, 1]
    ax4.plot(t_h, u_b, color=ORANGE, label="$u_n$")
    ax4.set_title("Heating up: applied voltage $u_n$")
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("voltage [V]")
    ax4.set_ylim(0, u_n_max * 1.15)
    ax4.legend(loc="lower right", fontsize=8.5)

    plt.show()