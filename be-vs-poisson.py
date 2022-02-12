# Bose-Einstein vs Poisson
#
# Dr Jesús Rubio
# University of Exeter
# J.Rubio-Jimenez@exeter.ac.uk
#
# Created: Feb 2022
# Last update: --

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Parameters
nu = 1
h = 1
kappa = 10
beta = 1

# Mean number
n_mean = nu**2 * kappa / (np.exp(beta * h * nu) - 1)

# Outcomes
n_outcomes = np.array([i for i in range(20)])

# Bose-Einstein distribution
bose_dist = np.exp(n_outcomes*np.log(n_mean / (1 + n_mean)) - np.log(1 + n_mean))

# Poisson distribution
poisson_dist = n_mean**n_outcomes * np.exp(-n_mean) / special.factorial(n_outcomes)

# Plots
plt.plot(n_outcomes, bose_dist, label = "Bose-Einstein dist.")
plt.plot(n_outcomes, poisson_dist, label = "Poisson dist.")
plt.xlabel("n")
plt.ylabel("p(n|...)")
plt.legend()
plt.grid(True)
plt.show()
