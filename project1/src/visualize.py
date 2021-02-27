import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_from_file

def brute_and_importance(fname_brute, fname_importance):
    alpha_brute, var_brute, exp_brute = read_from_file(fname_brute)
    alpha_importance, var_importance, exp_importance = read_from_file(fname_importance)

    fig = plt.figure()
    plt.grid()
    plt.plot(alpha_importance, exp_importance, color="k", label="importance")
    plt.plot(alpha_brute, exp_brute, color="tab:blue", label="brute force")
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"Energy")
    plt.legend()
    fig.savefig("../fig/compare_brute_importance.png")
    plt.show()

def gradient_descent(fname):
    alpha, var, exp = read_from_file(fname)
    plt.plot(alpha, exp, ".")
    plt.show()


def local_energy_alpha(fname, type):
    """
    temporary plot function
    """
    alpha, var_energy, exp_energy = read_from_file(fname)

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


if __name__ == "__main__":
    path = "generated_data"
    fname_brute_force = f"{path}/output_brute_force.txt"
    fname_importance = f"{path}/output_importance.txt"
    fname_gradient_descent = f"{path}/output_gradient_descent.txt"
    #brute_and_importance(fname_brute=fname_brute_force, fname_importance=fname_importance)
    #gradient_descent(fname_gradient_descent)

    f_importance = f"{path}/output_importance_particles.txt"
    f_brute_force = f"{path}/output_brute_force_particles.txt"
    # local_energy_alpha(f_brute_force, "brute_force")
    local_energy_alpha(f_importance, "importance")
