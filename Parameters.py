import numpy as np

# Motor values (both left and right are the same)
r_m = 6.37e-3 # radius
L_m = 0.3e-3 # inductance
R_m = 2.5  # resistance
J_m = 3e-6  # inertia
K_m = 0.04  # transducer constant

u_m_max = 24.0 # max voltage


m = 0.4 # tool head mass
m_b = 0.3 # bar mass

mu_r  = 0.05 # angular bearing friction coefficient
mu_lx = 1.0 # linear bearing friction in x direction
mu_ly = 1.5 # same in y direction

#Print area
d_x = 0.25 # bed size in x direction
d_y = 0.25 # bed size in y direction

# Nozzle
R_n = 14.4 # resistance
L_n = 2.2e-3 # inductance
C_n = 4.7e-6 # capacitance
u_n_max = 24.0 # max voltage
R_tn = 12.0 #thermal resistance
C_tn = 10.0 # thermal capacitance
T_n_ref = 210.0 # nozzle target
T_e = 25.0 # environments temperature
T_b = 60.0 #bed temperature

# Parameters for the state space
theta_l = None
theta_r = None
omega_l = None
omega_r = None
i_l = None
i_r = None
u_l = None
u_r = None