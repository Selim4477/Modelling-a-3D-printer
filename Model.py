import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Parameters import *

#import matplotlib.pyplot as plt

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

# Plotting the findings (AI assistance)
BLUE, ORANGE = "#2a78d6", "#eb6834"     # left motor / x  and  right motor / y
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": INK, "axes.edgecolor": "#b8b8b8",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
})

if time_array[-1] < 1:                   # short run: show milliseconds
    t_plot, t_label = time_array * 1e3, "time [ms]"
else:
    t_plot, t_label = time_array, "time [s]"
x_mm, y_mm = pos_x * 1e3, pos_y * 1e3    # position in millimetres
omega_l, omega_r = sol.y[2, :], sol.y[3, :]
i_l, i_r = sol.y[4, :], sol.y[5, :]

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
fig.suptitle(f"Constant input: $u_l$ = {u[0, 0]:g} V, $u_r$ = {u[1, 0]:g} V",
             fontsize=13, fontweight="bold", x=0.01, ha="left")

# (a) head position over time
a1 = ax[0, 0]
a1.plot(t_plot, x_mm, color=BLUE, label="x")
a1.plot(t_plot, y_mm, color=ORANGE, label="y")
a1.set_title("Head position")
a1.set_xlabel(t_label); a1.set_ylabel("position [mm]")
a1.legend(loc="upper left")

# (b) path of the head over the bed
a2 = ax[0, 1]
a2.add_patch(plt.Rectangle((0, 0), d_x * 1e3, d_y * 1e3, facecolor="#f2f2f2",
                           edgecolor="#9a9a9a", linewidth=1.2, zorder=0,
                           label="print bed"))
a2.plot(x_mm, y_mm, color=INK)
a2.plot(x_mm[0], y_mm[0], "o", color=INK, markersize=7, label="start")
a2.plot(x_mm[-1], y_mm[-1], "o", markerfacecolor="white", markeredgecolor=INK,
        markeredgewidth=2, markersize=8, label="end")
a2.set_title("Path of the head")
a2.set_xlabel("x [mm]"); a2.set_ylabel("y [mm]")
a2.set_aspect("equal", adjustable="datalim")
a2.legend(loc="upper right")

# (c) motor speeds
a3 = ax[1, 0]
a3.plot(t_plot, omega_l, color=BLUE, label=r"$\omega_l$ (left)")
a3.plot(t_plot, omega_r, color=ORANGE, label=r"$\omega_r$ (right)")
a3.axhline(0, color="#b8b8b8", linewidth=1)
a3.set_title("Motor speeds")
a3.set_xlabel(t_label); a3.set_ylabel("angular speed [rad/s]")
a3.legend(loc="best")

# (d) motor currents
a4 = ax[1, 1]
a4.plot(t_plot, i_l, color=BLUE, label=r"$i_l$ (left)")
a4.plot(t_plot, i_r, color=ORANGE, label=r"$i_r$ (right)")
a4.axhline(0, color="#b8b8b8", linewidth=1)
a4.set_title("Motor currents")
a4.set_xlabel(t_label); a4.set_ylabel("current [A]")
a4.legend(loc="best")

plt.show()