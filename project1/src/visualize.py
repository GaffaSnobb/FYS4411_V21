import os
from matplotlib import interactive
import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_from_file, read_energy_from_file, read_all_files

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
    plt.ylabel(r"$E_{L} \ / \ N$")
    #plt.ylabel(r"$\frac{E_{L}}{N}$")
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
    plt.ylabel(r"$E_{L} \ / \ N$")
    plt.legend()
    plt.show()


def local_energy_alpha(fname, method):
    """
    temporary plot function
    """
    alpha, var_energy, exp_energy, time = read_from_file(fname)

    #exp_energy /= n_particles
    #var_energy /= n_particles

    fig = plt.figure()
    plt.grid()
    plt.title(method)
    plt.plot(alpha, exp_energy, color="k", label="Expected local energy")
    plt.fill_between(
        alpha,
        exp_energy - np.sqrt(var_energy),
        exp_energy + np.sqrt(var_energy),
        color="k",
        alpha=0.2,
        label="std"
    )
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"Local Energy, $E_{L}$")
    plt.legend()
    fig.savefig(f"../fig/VMC_{method}_variance.png")
    plt.show()


def tmp():
    f_energy = "generated_data/output_energy_gradient_descent.txt"
    alpha, energy, n_particles = read_energy_from_file(f_energy)

    data = energy[:,0]
    iter = np.arange(len(data))
    plt.plot(iter, data)
    plt.show()


def onebody(fname):
    alphas = np.loadtxt(fname, max_rows=1)
    data = np.loadtxt(fname, skiprows=1)
    bins = np.arange(0, data.shape[0], 1)

    print(f"{alphas=}")
    plt.bar(bins, data[:, 4]/np.trapz(data[:, 4])) # Halfway.
    plt.bar(bins, data[:, -1]/np.trapz(data[:, -1]), alpha=0.5)
    plt.xlabel("bins")
    plt.ylabel("scaled counts")
    plt.show()


def task_1b():
    brute = read_all_files(filter_method="brute", filter_data_type="particles")
    importance = read_all_files(filter_method="importance", filter_data_type="particles")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    ax = ax.ravel()
    # ax[0].set_xlabel(r"$\alpha$")
    # ax[0].set_ylabel("Energy per particle")
    fig.text(x=0.5, y=0.01, s=r"$\alpha$", fontsize=15)
    fig.text(x=0.005, y=0.35, s=r"Energy per particle", fontsize=15, rotation="vertical")

    ax[0].plot(
        brute[0].data[:, 0],
        brute[0].data[:, 2],
        label = f"Brute"
    )
    ax[0].plot(
        importance[0].data[:, 0],
        importance[0].data[:, 2],
        label = f"Importance"
    )


    # Place legend in ax[1].
    handles, labels = ax[0].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc='center')
    ax[1].axis("off")

    ax[2].plot(
        brute[1].data[:, 0],
        brute[1].data[:, 2],
        label = f"Brute\nn particles: {brute[1].n_particles}"
    )
    ax[2].plot(
        importance[1].data[:, 0],
        importance[1].data[:, 2],
        label = f"Importance\nn particles: {importance[1].n_particles}"
    )
    ax[3].plot(
        brute[2].data[:, 0],
        brute[2].data[:, 2],
        label = f"Brute\nn particles: {brute[2].n_particles}"
    )

    for i in range(len(ax)):
        ax[i].tick_params(labelsize=13)

    fig.tight_layout(pad=2)
    plt.show()








if __name__ == "__main__":
    # path = "generated_data"
    # fname_brute_force = f"{path}/output_brute_force.txt"
    # fname_importance = f"{path}/output_importance.txt"
    # fname_gradient_descent = f"{path}/output_gradient_descent.txt"
    #brute_and_importance(fname_brute=fname_brute_force, fname_importance=fname_importance)
    #gradient_descent(fname_gradient_descent)

    # f_importance = f"{path}/output_importance_particles.txt"
    # f_brute_force = f"{path}/output_brute_force_particles.txt"
    # f_brute_force_onebody = f"{path}/output_brute_force_onebody_density.txt"
    # local_energy_alpha(f_brute_force, "brute_force")
    # local_energy_alpha(f_importance, "importance")
    # local_energy_alpha(f"{path}/output_gradient_descent_particles.txt", "GD")
    # tmp_gd()
    # onebody(f_brute_force_onebody)
    task_1b()