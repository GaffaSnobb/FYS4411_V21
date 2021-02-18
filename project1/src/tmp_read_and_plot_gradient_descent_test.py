import numpy as np
import matplotlib.pyplot as plt

alpha, expected_energy = np.loadtxt("generated_data/gradient_descent_test.txt", unpack=True, skiprows=1)

plt.plot(alpha[:-1], expected_energy[:-1], ".")
plt.show()