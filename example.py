# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 22:30:12 2024

@author: 29639
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Lorenz system
def tree1(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def tree2(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def env(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial state
initial_state = [1.0, 1.0, 1.0]

# Time span for the solution
t_span = (0, 50)
t_eval = np.linspace(*t_span, 5000)

# Solve the ODE
solution = solve_ivp(
    lorenz,
    t_span,
    initial_state,
    t_eval=t_eval,
    args=(sigma, rho, beta),
)

# Extract the solution
x, y, z = solution.y

# Plotting the Lorenz attractor
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5, color='blue')
ax.set_title("Lorenz Attractor", fontsize=14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
