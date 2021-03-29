import sys
import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_from_file, read_energy_from_file, read_all_files
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]) 

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
    brute_3d = read_all_files(
        filter_method = "brute",
        filter_n_particles = None,
        filter_n_dims = 3,
        filter_n_mc_cycles = int(2**20),
        filter_step_size = None,
        filter_numerical = False,
        filter_interaction = False,
        filter_data_type = "particles"
    )
    brute_2d = read_all_files(
        filter_method = "brute",
        filter_n_particles = None,
        filter_n_dims = 2,
        filter_n_mc_cycles = int(2**20),
        filter_step_size = None,
        filter_numerical = False,
        filter_interaction = False,
        filter_data_type = "particles"
    )
    brute_1d = read_all_files(
        filter_method = "brute",
        filter_n_particles = None,
        filter_n_dims = 1,
        filter_n_mc_cycles = int(2**20),
        filter_step_size = None,
        filter_numerical = False,
        filter_interaction = False,
        filter_data_type = "particles"
    )
    importance_3d = read_all_files(
        filter_method = "importance",
        filter_n_particles = None,
        filter_n_dims = 3,
        filter_n_mc_cycles = int(2**20),
        filter_step_size = None,
        filter_numerical = False,
        filter_interaction = False,
        filter_data_type = "particles"
    )
    importance_2d = read_all_files(
        filter_method = "importance",
        filter_n_particles = None,
        filter_n_dims = 2,
        filter_n_mc_cycles = int(2**20),
        filter_step_size = None,
        filter_numerical = False,
        filter_interaction = False,
        filter_data_type = "particles"
    )
    importance_1d = read_all_files(
        filter_method = "importance",
        filter_n_particles = None,
        filter_n_dims = 1,
        filter_n_mc_cycles = int(2**20),
        filter_step_size = None,
        filter_numerical = False,
        filter_interaction = False,
        filter_data_type = "particles"
    )
    
    def one_plot(data_3d, data_2d, data_1d):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))

        ax.errorbar(
            data_3d.data[:, 0],
            data_3d.data[:, 2],
            # np.sqrt(data_3d.data[:, 1]),
            # fmt = "--o",
            label = f"3D",
            # color = "black",
            capsize = 3
        )
        ax.errorbar(
            data_2d.data[:, 0],
            data_2d.data[:, 2],
            # np.sqrt(data_2d.data[:, 1]),
            # fmt = "--D",
            label = f"2D",
            # color = "black",
            capsize = 3
        )
        ax.errorbar(
            data_1d.data[:, 0],
            data_1d.data[:, 2],
            # np.sqrt(data_1d.data[:, 1]),
            # fmt = "--s",
            label = f"1D",
            # color = "black",
            capsize = 3
        )
        ax.tick_params(labelsize=13)
        ax.grid()
        ax.legend(fontsize=15)
        ax.set_xticks(np.arange(0.1, 1 + 0.1, 0.1))
        fig.text(x=0.5, y=0.01, s=r"$\alpha$", fontsize=15)
        fig.text(x=0.005, y=0.42, s=r"Local energy", fontsize=15, rotation="vertical")
        fig.tight_layout(pad=2)
        
        fname_out_lst = data_3d.fname.split("_")
        fname_out_lst.pop(0)
        fname_out_lst.pop(2)
        fname_out_lst.pop(-1)
        fname_out_lst.pop(-1)
        fname_out = ""
        
        for elem in fname_out_lst:
            fname_out += elem
            fname_out += "_"
        fname_out += "dimension_plot.png"
        
        fig.savefig(fname = "../fig/" + fname_out, dpi=300)
        plt.show()

    one_plot(   # 10 particles.
        data_3d = importance_3d[0],
        data_2d = importance_2d[0],
        data_1d = importance_1d[0],
    )
    one_plot(   # 100 particles.
        data_3d = importance_3d[1],
        data_2d = importance_2d[1],
        data_1d = importance_1d[1],
    )
    one_plot(   # 10 particles.
        data_3d = brute_3d[0],
        data_2d = brute_2d[0],
        data_1d = brute_1d[0],
    )


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