import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_energy_from_file
from time import time

def block(x):
    """
    Credit: Marius Jonsson
    Jonsson, M. (2018). Standard error estimation by an automated blocking method. Physical Review E, 98(4), 043304.
    """
    # preliminaries
    n = len(x)
    d = int(np.log2(n))
    s, gamma, iter = np.zeros(d), np.zeros(d), np.zeros(d)
    mu = np.mean(x)
    t0 = time()

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in np.arange(0, d):
        iter[i] = i
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*np.sum( (x[0:(n-1)]-mu)*(x[1:n]-mu) )

        # estimate variance of x
        s[i] = np.var(x)

        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (np.cumsum( ((gamma/s)**2*2**np.arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    q = np.array([6.634897,9.210340, 11.344867, 13.276704, 15.086272, 16.811894,
                18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967,
                27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306,
                36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820,
                44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in np.arange(0,d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        print ("Warning: Use more data")

    ans = s[k]/2**(d-k)
    print(f"avg: {mu:.3f}, iterations: {k}, std. {ans**.5:.3f}\n")
    return ans



"""
outline:
Need energy array (n_alphas, n_mc) or smt.

Need list of alpha values (can be read from file?)
alphas = np.array(n_alphas)

loop over alpha values and do blocking

for this alpha value:
    call block function with x = energies[this alpha, :]

"""

file = "generated_data/output_energy_importance.txt"
alpha, energy = read_energy_from_file(file)
print(45*"_", "\n")
print("Importance")
print(45*"_", "\n")

for i in range(len(alpha)):
    print(f"alpha: {alpha[i]:.2f}")
    ans = block(energy[:,i])

file = "generated_data/output_energy_brute_force.txt"
alpha, energy = read_energy_from_file(file)
print(45*"_", "\n")
print("Brute force")
print(45*"_", "\n")

for i in range(len(alpha)):
    print(f"alpha: {alpha[i]:.2f}")
    ans = block(energy[:,i])
