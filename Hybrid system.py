# Imports
import numpy as np
import os
from Parameters import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Model import A_x, B_x, A_y, B_y, M_x, M_y, b_x, b_y
from Model import timing, derivatives, inversion
from Model import x_c, y_c, a_e, b_e
from Model import nozzle, u_hold


# ------------------------------ Printer settings/choices (all in SI units except for the temperatures) ----------------------------
t_lap = 4.0          # time to print one layer
t_move = 1.5         # time for a travel move
t_wiped = 3.0        # time to wipe off the excess filament
t_load = 15.0        # time the user needs to load the new filament
t_collect = 5.0      # time the user needs to take the print
T_high = T_n_ref     # nozzle hot enough to print
T_low = 90.0         # nozzle cold enough to change the filament
U = 1.0              # voltage of the motion commands V and H
eps = 0.005          # lower limit for ang speed of the motor
tau_min = 0.2        # skip the start of a move, when the motors are still speeding up
k_w = 1e8            # stiffness to constrain drawing area
c_w = 2e5            # damping to constrain drawing area

colours = ["A", "B"]         # 2 layers, different colors
spools = [0.03, 1.0]         # filament on the spools

# Wall physics
def wall(p, v, p_m):
    if p < 0:
        return max(0.0, -k_w*p - c_w*v)
    if p > p_m:
        return min(0.0, -k_w*(p - p_m) - c_w*v)
    return 0.0

# The system equations with the wall force + filament left on the spool added
def sys(x, u_x, u_y, u_n, extrude):
    dx_axis = A_x @ x[0:3] + B_x[:, 0]*u_x
    dy_axis = A_y @ x[3:6] + B_y[:, 0]*u_y
    dx_axis[1] = dx_axis[1] + wall(x[0], x[1], d_x)/M_x
    dy_axis[1] = dy_axis[1] + wall(x[3], x[4], d_y)/M_y
    d_nozzle = nozzle(0, x[6:9], u_n)
    d_tau = 1.0
    if extrude:
        d_e = -np.sqrt(x[1]**2 + x[4]**2)             # filament used = distance printed
    else:
        d_e = 0.0
    return np.concatenate([dx_axis, dy_axis, d_nozzle, [d_tau, d_e]])

def vot(X, Y, t):
    vel, acc, jerk = derivatives(X, t)
    u_x, i_x = inversion(vel, acc, jerk, M_x, b_x)
    vel, acc, jerk = derivatives(Y, t)
    u_y, i_y = inversion(vel, acc, jerk, M_y, b_y)
    return t, u_x, u_y

def elps(phi_0):
    t, p = timing(t_lap*(1 - phi_0/(2*np.pi)))
    phi = phi_0 + (1 - phi_0/(2*np.pi))*p
    X = x_c + a_e*np.cos(phi)
    Y = y_c + b_e*np.sin(phi)
    return vot(X, Y, t)

def line(x0, y0, x1, y1):
    t, p = timing(t_move)
    s = p/(2*np.pi)
    X = x0 + (x1 - x0)*s
    Y = y0 + (y1 - y0)*s
    return vot(X, Y, t)

def inputs(x):
    tau = x[9]
    # motor voltages
    if mode == "Printing" or mode == "Return" or mode == "Collection":
        if tau < plan_t[-1]:
            u_x = np.interp(tau, plan_t, plan_ux)
            u_y = np.interp(tau, plan_t, plan_uy)
        else:
            u_x = 0.0
            u_y = 0.0
    elif mode == "Park V-" or mode == "Down V-":
        u_x = 0.0
        u_y = -U
    elif mode == "Park H-" or mode == "Left H-":
        u_x = -U
        u_y = 0.0
    elif mode == "Up V+":
        u_x = 0.0
        u_y = U
    elif mode == "Right H+":
        u_x = U
        u_y = 0.0
    else:
        u_x = 0.0
        u_y = 0.0
    # heater voltage
    if mode == "Prepare nozzle" or mode == "Filament replaced":
        u_n = u_n_max                                 # heat up
    elif mode == "Idle" or mode == "Cooling" or mode == "Loading" or mode == "Collection":
        u_n = 0.0                                     # heater off
    else:
        u_n = u_hold                                  # keep the nozzle hot
    # extruding or not
    if mode == "Printing" or mode == "Up V+" or mode == "Right H+" or mode == "Down V-" or mode == "Left H-":
        extrude = True
    else:
        extrude = False
    return u_x, u_y, u_n, extrude

def f(t, x):
    u_x, u_y, u_n, extrude = inputs(x)
    return sys(x, u_x, u_y, u_n, extrude)

def guard(t, x):
    tau = x[9]
    w_l = (x[1] + x[4])/r_m
    w_r = (x[1] - x[4])/r_m
    if mode == "Idle":
        return tau - 1.0
    elif mode == "Prepare nozzle" or mode == "Filament replaced":
        return x[6] - T_high
    elif mode == "Printing":
        return max(tau - plan_t[-1], -x[10])
    elif mode == "Excess filament":
        return tau - t_wiped
    elif mode == "Cooling":
        return T_low - x[6]
    elif mode == "Loading":
        return tau - t_load
    elif mode == "Return":
        return tau - t_move
    elif mode == "Collection":
        return tau - t_collect
    else:
        return min(tau - tau_min, eps - max(abs(w_l), abs(w_r)))
guard.terminal = True

for job in ["print", "square"]:
    # initial state
    x = np.zeros(11)
    x[6] = T_e
    if job == "print":
        x[0] = x_c + a_e
        x[3] = y_c
        x[10] = 1.0
    else:
        x[0] = 0.155
        x[3] = 0.125
        x[10] = 2.0
    mode = "Idle"
    t = 0.0
    n = 0
    colour = "A"
    spool = 0
    plan_t = [0.0, 1.0]
    plan_ux = [0.0, 0.0]
    plan_uy = [0.0, 0.0]
    log = []
    print("Job:", job)

    while True:

        sol = solve_ivp(f, (t, 3000), x, events=guard, method="LSODA",
                        max_step=0.05, rtol=1e-6, atol=1e-9)
        log.append([mode, colour, sol.t, sol.y])
        if sol.status != 1:
            break
        t = sol.t[-1]
        x = sol.y[:, -1].copy()
        X = x[0]
        Y = x[3]
        old_mode = mode
        x[9] = 0.0
        if mode == "Idle":
            mode = "Prepare nozzle"
        elif mode == "Prepare nozzle":
            if job == "print":
                n = 1
                plan_t, plan_ux, plan_uy = elps(0)
                mode = "Printing"
            else:
                mode = "Park V-"
        elif mode == "Printing":
            if x[10] < 1e-6:
                x_s = X
                y_s = Y
                x_F = X
                y_F = Y
                phi_s = np.arctan2((Y - y_c)/b_e, (X - x_c)/a_e)
                if phi_s < 0:
                    phi_s = phi_s + 2*np.pi
                mode = "Change spool"
            elif n == len(colours):
                plan_t, plan_ux, plan_uy = line(X, Y, 0.0, 0.0)
                mode = "Collection"
            else:
                n = n + 1
                if colours[n-1] != colour:
                    x_s = X
                    y_s = Y
                    phi_s = 0.0
                    mode = "Change spool"
                else:
                    plan_t, plan_ux, plan_uy = elps(0)
        elif mode == "Change spool":
            mode = "Park V-"
        elif mode == "Park V-":
            mode = "Park H-"
        elif mode == "Park H-":
            if job == "print":
                mode = "Excess filament"
            else:
                mode = "Up V+"
        elif mode == "Excess filament":
            mode = "Cooling"
        elif mode == "Cooling":
            mode = "Loading"
        elif mode == "Loading":
            x[10] = spools[spool]
            spool = spool + 1
            colour = colours[n-1]
            mode = "Filament replaced"
        elif mode == "Filament replaced":
            plan_t, plan_ux, plan_uy = line(X, Y, x_s, y_s)
            mode = "Return"
        elif mode == "Return":
            plan_t, plan_ux, plan_uy = elps(phi_s)
            mode = "Printing"
        elif mode == "Collection":
            mode = "Idle"
        elif mode == "Up V+":
            mode = "Right H+"
        elif mode == "Right H+":
            mode = "Down V-"
        elif mode == "Down V-":
            mode = "Left H-"
        elif mode == "Left H-":
            mode = "Idle"

        print("t =", round(t, 2), "s:", old_mode, "->", mode)
        if mode == "Idle":
            break

    if job == "print":
        log31 = log
        x_stop = x_F
        y_stop = y_F
    else:
        log32 = log

# ------------------------------- Plots (AI assistance) -------------------------------
plt.rcParams["axes.grid"] = True

def plot_modes(ax, log, names):

    times = []
    numbers = []
    for piece in log:
        times.append(piece[2][0])
        numbers.append(names.index(piece[0]))
    times.append(log[-1][2][-1])
    numbers.append(numbers[-1])
    ax.step(times, numbers, where="post", color="black")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_title("Mode")
    ax.set_xlabel("time [s]")

def plot_sensors(ax, pieces):

    for piece in pieces:
        w_l = (piece[3][1] + piece[3][4])/r_m
        w_r = (piece[3][1] - piece[3][4])/r_m
        ax.plot(piece[2], w_l, color="tab:blue")
        ax.plot(piece[2], w_r, color="tab:orange")
        ax.axvline(piece[2][-1], color="gray", linestyle=":")
    ax.legend([r"$\omega_l$", r"$\omega_r$"])
    ax.set_title("Angular motion sensors (switch when both are 0)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angular speed [rad/s]")

names31 = ["Idle", "Prepare nozzle", "Printing", "Change spool", "Park V-", "Park H-",
           "Excess filament", "Cooling", "Loading", "Filament replaced", "Return", "Collection"]
fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle("Q3.1 Filament replacement")
plot_modes(ax[0], log31, names31)
for piece in log31:
    ax[1].plot(piece[2], piece[3][6], color="tab:red")
    ax[2].plot(piece[2], piece[3][10], color="black")
ax[1].axhline(T_high, linestyle="--", color="gray")
ax[1].axhline(T_low, linestyle="--", color="gray")
ax[1].set_title("Nozzle temperature (dashed: $T_{high}$ and $T_{low}$)")
ax[1].set_ylabel("$T_n$ [C]")
ax[2].set_title("Filament left on the spool (F when it reaches 0)")
ax[2].set_ylabel("e [m]")
ax[2].set_xlabel("time [s]")
fig.tight_layout()
fig.savefig("q31_modes.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Q3.1 Motion of the head")
park = []
for piece in log31:
    X = piece[3][0]*1000
    Y = piece[3][3]*1000
    if piece[0] == "Printing" and piece[1] == "A":
        ax[0].plot(X, Y, color="tab:blue", linewidth=5)
    elif piece[0] == "Printing":
        ax[0].plot(X, Y, color="tab:orange", linewidth=2, zorder=3)
    elif piece[0] == "Change spool" or piece[0] == "Return" or piece[0] == "Collection":
        ax[0].plot(X, Y, color="gray", linestyle="--", linewidth=1)
    elif piece[0] == "Park V-" or piece[0] == "Park H-":
        ax[0].plot(X, Y, color="gray", linestyle="--", linewidth=1)
        if len(park) < 2:
            park.append(piece)
ax[0].plot([], [], color="tab:blue", linewidth=5, label="colour A")
ax[0].plot([], [], color="tab:orange", linewidth=2, label="colour B")
ax[0].plot([], [], color="gray", linestyle="--", label="travel / parking")
ax[0].plot(0, 0, "ks", label="wipe tower")
ax[0].set_title("Path of the head")
ax[0].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
ax[0].axis("equal")
ax[0].legend()
plot_sensors(ax[1], park)
fig.tight_layout()
fig.savefig("q31_motion.png", dpi=150)


names32 = ["Idle", "Prepare nozzle", "Park V-", "Park H-", "Up V+", "Right H+", "Down V-", "Left H-"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Q3.2 Drawing a square with only the angular motion sensors")
moving = []
for piece in log32:
    X = piece[3][0]*1000
    Y = piece[3][3]*1000
    if piece[0] == "Park V-" or piece[0] == "Park H-":
        ax[0].plot(X, Y, color="gray", linestyle="--")
        moving.append(piece)
    elif piece[0] != "Idle" and piece[0] != "Prepare nozzle":
        ax[0].plot(X, Y, color="tab:blue", linewidth=3)
        moving.append(piece)
ax[0].set_title("Path (dashed: parking, blue: printed)")
ax[0].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
ax[0].axis("equal")
plot_sensors(ax[1], moving)
plot_modes(ax[2], log32, names32)
ax[2].set_xlim(moving[0][2][0] - 0.5, moving[-1][2][-1] + 0.5)     # skip the heating
fig.tight_layout()
fig.savefig("q32_square.png", dpi=150)
plt.show()


