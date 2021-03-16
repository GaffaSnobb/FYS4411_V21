import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_from_file

def brute_and_importance(fname_brute, fname_importance):
    alpha_brute, var_brute, exp_brute, n_brute = read_from_file(fname_brute)
    alpha_impor, var_impor, exp_impor, n_impor = read_from_file(fname_importance)

    # Get energy per particle
    exp_brute /= n_brute
    var_brute /= n_brute
    exp_impor /= n_impor
    var_impor /= n_impor

    fig = plt.figure()
    plt.grid()
    plt.errorbar(alpha_impor, exp_impor, np.sqrt(var_impor), fmt=".", color="tab:red", label="importance")
    plt.errorbar(alpha_brute, exp_brute, np.sqrt(var_brute), fmt=".", color="tab:blue", label="brute force")
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"E/N")
    plt.legend()
    fig.savefig("../fig/compare_brute_importance.png")
    plt.show()


def gradient_descent(fname):
    alpha, var, exp, n_particles = read_from_file(fname)

    var /= n_particles
    exp /= n_particles

    fig = plt.figure()
    plt.grid()
    plt.errorbar(alpha, exp, np.sqrt(var), fmt=".", color="tab:red", label="Gradient descent")
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"E/N")
    plt.legend()
    plt.show()


def local_energy_alpha(fname, type):
    """
    temporary plot function
    """
    alpha, var_energy, exp_energy, n_particles = read_from_file(fname)

    exp_energy /= n_particles
    var_energy /= n_particles

    fig = plt.figure()
    plt.grid()
    plt.title(type)
    plt.plot(alpha, exp_energy, color="k", label="Expected local energy")
    plt.fill_between(alpha, exp_energy - np.sqrt(var_energy), exp_energy + np.sqrt(var_energy), color="k", alpha=0.2, label="std")
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"E/N")
    plt.legend()
    fig.savefig(f"../fig/VMC_{type}_variance.png")
    plt.show()


def tmp_gd():
    """
    temporary plot function
    """
    alpha, var_energy, exp_energy = read_from_file("generated_data/output_gradient_descent_particles.txt")

    fig = plt.figure()
    plt.grid()
    plt.plot(alpha, exp_energy, ".", color="k", label="Expected local energy")
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"E/N")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    path = "generated_data"
    fname_brute_force = f"{path}/output_brute_force.txt"
    fname_importance = f"{path}/output_importance.txt"
    fname_gradient_descent = f"{path}/output_gradient_descent.txt"

    f_importance = f"{path}/output_importance_particles.txt"
    f_brute_force = f"{path}/output_brute_force_particles.txt"
    # local_energy_alpha(f_brute_force, "brute_force")
    # local_energy_alpha(f_importance, "importance")
    # local_energy_alpha(f"{path}/output_gradient_descent_particles.txt", "GD")
    tmp_gd()
