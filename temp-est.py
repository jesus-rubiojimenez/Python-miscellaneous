# Example of data analysis in scale-invariant thermometry 
#
# Dr Jesús Rubio
# University of Exeter
#
# Created: Feb 2022
# Modified: --

import numpy as np
from scipy import special
import matplotlib.pyplot as plt

# Data (change accordingly)
data = np.loadtxt("dummy_data.txt", dtype=float)
mu = np.size(data)
h = 1
nu = 1
kB = 1
kappa = 1
Treal = 1

# Hypothesis space
thetamin = 10**3
thetamax = 10**4
theta = np.linspace(thetamin,thetamax,10**4)
nmean = nu**2 * kappa / (np.exp(h * nu / (kB*theta)) - 1)

# Inference
prob_temp = 1 / (np.trapz(1 / theta, x = theta) * theta) # prior
optEst = np.zeros(mu)
optErr = np.zeros(mu)
for runs in range(1, mu + 1):
        
    # Likelihood, joint, evidence and posterior functions
    likelihood = np.exp(-nmean + data[runs-1] * np.log(nmean) - np.log(special.gamma(data[runs-1] + 1))) # likelihood function
    joint = prob_temp * likelihood # joint probability
    evidence = np.trapz(joint, x = theta) # normalisation of Bayes theorem
    prob_temp = joint / evidence # posterior probability
                
    # Optimal estimator
    aux = prob_temp * np.log(theta)
    optLogEst = np.trapz(aux, x = theta)
    np.put(optEst, runs-1, np.exp(optLogEst), mode='raise')
    
    # Optimal uncertainty
    np.put(optErr, runs-1, np.trapz(aux * np.log(theta), x = theta) - optLogEst**2, mode='raise')

optErrBar = optEst * np.sqrt(optErr)
mu_range = np.array(range(1, mu + 1))

# Plots
plt.plot(mu_range, optEst, label = "Estimate")
plt.plot(mu_range, Treal * np.ones(mu), label = "True value")
plt.fill_between(mu_range, optEst-optErrBar, optEst+optErrBar, alpha=0.5, antialiased=True)

plt.xlabel("Number of data")
plt.ylabel("Temperature")

ax = plt.gca()
ax.set_xlim([1, mu])
ax.set_ylim([Treal - np.max(np.abs(optErrBar)) / 2, Treal + np.max(np.abs(optErrBar)) / 2])

plt.legend()
plt.grid(True)
plt.show()
