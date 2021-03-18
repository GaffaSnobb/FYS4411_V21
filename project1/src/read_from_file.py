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

def read_energy_from_file(filename, clip = False):
    """
    for files with names: output_energy_*.txt
    """
    n_particles = get_number_particles(filename)
    data = np.loadtxt(filename, skiprows=1)
    alphas = data[0,:]
    energies = data[1:,:]


    if clip:
        first_zero_elm = first_zero(energies[0,:], axis=0)
        if first_zero_elm < 0:
            pass
        else:
            new_alphas = alphas[:first_zero_elm]
            new_energies = energies[:,:first_zero_elm]

            energies = new_energies
            alphas = new_alphas
    return alphas, energies, n_particles

def first_zero(arr, axis, invalid_val=-1):
    mask = arr==0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


if __name__ == "__main__":
    #filename = "generated_data/output_energy_gradient_descent_test.txt"
    filename = "generated_data/output_energy_gradient_descent.txt"
    alphas, energies1, n_particles = read_energy_from_file(filename)
    alphas, energies, n_particles = read_energy_from_file(filename, clip=True)

    print(energies, energies1, sep="\n")
