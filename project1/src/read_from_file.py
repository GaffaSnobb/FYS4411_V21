import numpy as np
import matplotlib.pyplot as plt

def get_number_particles(filename):
    f = open(filename)
    n_particles = (f.readline()).split()[1]
    f.close()
    return float(n_particles)

def read_from_file(filename):
    n_particles = get_number_particles(filename)
    alpha, var, exp = np.loadtxt(fname=filename, skiprows=2, unpack=True)
    return alpha, var, exp, n_particles

def read_energy_from_file(filename):
    """
    for files with names: output_energy_*.txt
    """
    n_particles = get_number_particles(filename)
    data = np.loadtxt(filename, skiprows=1)
    alpha = data[0,:]
    energy = data[1:,:]
    return alpha, energy, n_particles

if __name__ == "__main__":
    filename = "generated_data/output_gradient_descent.txt"
