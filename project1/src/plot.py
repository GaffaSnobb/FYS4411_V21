import sys
import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_all_files
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:green", "tab:purple","tab:red", "tab:orange", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
mpl.rcParams.update({'font.size': 15})

def read_all_dimensions(method, n_particles, n_mc_cycles, step_size, numerical, interaction, data_type):
    """
    --------------
    --------------
    """
    input_1d = read_all_files(
        filter_method = method,
        filter_n_particles = n_particles,
        filter_n_dims = 1,
        filter_n_mc_cycles = n_mc_cycles,
        filter_step_size = step_size,
        filter_numerical = numerical,
        filter_interaction = interaction,
        filter_data_type = data_type)

    input_2d = read_all_files(
        filter_method = method,
        filter_n_particles = n_particles,
        filter_n_dims = 2,
        filter_n_mc_cycles = n_mc_cycles,
        filter_step_size = step_size,
        filter_numerical = numerical,
        filter_interaction = interaction,
        filter_data_type = data_type)

    input_3d = read_all_files(
        filter_method = method,
        filter_n_particles = n_particles,
        filter_n_dims = 3,
        filter_n_mc_cycles = n_mc_cycles,
        filter_step_size = step_size,
        filter_numerical = numerical,
        filter_interaction = interaction,
        filter_data_type = data_type)


    return input_1d[0], input_2d[0], input_3d[0]


def plot_energy(method, n_particles, n_mc_cycles, step_size, numerical, interaction):

    label_nd = ""
    if method == "brute": label_nd = "Brute-force"
    if method == "importance": label_nd = "Importance"

    data_type   = "particles"

    data_1d, data_2d, data_3d = read_all_dimensions(method, n_particles,
                                                    n_mc_cycles, step_size,
                                                    numerical, interaction,
                                                    data_type)


    fig = plt.figure(figsize=(9, 7))
    plt.grid()

    plt.plot(data_1d.data[:, 0], data_1d.data[:, 2], label=label_nd+", 1D")
    plt.fill_between(data_1d.data[:,0],
                     data_1d.data[:,2] - np.sqrt(data_1d.data[:, 1]),
                     data_1d.data[:,2] + np.sqrt(data_1d.data[:, 1]),
                     color="k",
                     alpha=0.2,
                     label=r"Standard deviation, $\sigma$")

    plt.plot(data_2d.data[:, 0], data_2d.data[:, 2], label=label_nd+", 2D")
    plt.fill_between(data_2d.data[:,0],
                     data_2d.data[:,2] - np.sqrt(data_2d.data[:, 1]),
                     data_2d.data[:,2] + np.sqrt(data_2d.data[:, 1]),
                     color="k",
                     alpha=0.2)

    plt.plot(data_3d.data[:, 0], data_3d.data[:, 2], label=label_nd+", 3D")
    plt.fill_between(data_3d.data[:,0],
                     data_3d.data[:,2] - np.sqrt(data_3d.data[:, 1]),
                     data_3d.data[:,2] + np.sqrt(data_3d.data[:, 1]),
                     color="k",
                     alpha=0.2)


    plt.xticks(np.arange(0.1, 1 + 0.1, 0.1))
    plt.xlabel(r"$ \alpha $")
    plt.ylabel(r"Local Energy, $E_{L}$")
    plt.legend()

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
    fig.tight_layout(pad=2)
    fig.savefig(fname = "../fig/" + fname_out, dpi=300)
    plt.show()



def plot_variance_MC(cycle_list, n_particles, n_dims, step_size_brute, step_size_importance, numerical, interaction, alpha_place):

    variances_brute = []
    variances_importance = []

    for n_mc_cycles_exponent in cycle_list:

        n_mc_cycles = 2**n_mc_cycles_exponent

        brute = read_all_files(
            filter_method = "brute",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_brute,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "particles")[0]

        importance = read_all_files(
            filter_method = "importance",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_importance,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "particles")[0]

        variances_brute.append((brute.data[:, 1][alpha_place]/n_mc_cycles)**0.5)
        variances_importance.append((importance.data[:, 1][alpha_place]/n_mc_cycles)**0.5)


    variances_brute = np.array(variances_brute)
    variances_importance = np.array(variances_importance)

    mc_cycles = np.array(cycle_list)

    fig = plt.figure(figsize=(9, 7))
    plt.grid()
    plt.plot(mc_cycles, variances_brute, label = "Brute-force")
    plt.plot(mc_cycles, variances_importance, label= "Importance")
    plt.legend()
    plt.xticks(mc_cycles)
    plt.xlabel(r"Number of MC-cycles")
    plt.ylabel(r"Standard deviation,  $\sigma$")

    fname_importance = importance.fname.split("_")
    fname_brute = brute.fname.split("_")

    fname_out = f"brute_importance_{n_particles}_{n_dims}_"
    fname_out += f"{step_size_importance}_{step_size_brute}_"
    fname_out += f"{fname_brute[6]}_{fname_brute[7]}_"
    fname_out += "error_mc_plot.png"
    fig.tight_layout(pad=2)
    fig.savefig(fname = "../fig/" + fname_out, dpi=300)

    plt.show()

def plot_energy_MC_old(cycle_list, n_particles, n_dims, step_size_brute, step_size_importance, numerical, interaction, alpha_place):
    # uinteressant
    energies_brute = []
    energies_importance = []

    for n_mc_cycles_exponent in cycle_list:

        n_mc_cycles = 2**n_mc_cycles_exponent

        brute = read_all_files(
            filter_method = "brute",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_brute,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "particles")[0]

        importance = read_all_files(
            filter_method = "importance",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_importance,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "particles")[0]

        energies_brute.append(brute.data[:, 2][alpha_place])
        energies_importance.append(importance.data[:, 2][alpha_place])


    energies_brute = np.array(energies_brute)
    energies_importance = np.array(energies_importance)

    mc_cycles = np.array(cycle_list)

    fig = plt.figure(figsize=(9, 7))
    plt.grid()
    plt.plot(mc_cycles, energies_brute, label = "Brute-force")
    plt.plot(mc_cycles, energies_importance, label= "Importance")
    plt.legend()
    plt.xticks(mc_cycles)
    plt.xlabel(r"Number of MC-cycles")
    plt.ylabel(r"Expected energy")

    fname_importance = importance.fname.split("_")
    fname_brute = brute.fname.split("_")

    fname_out = f"brute_importance_{n_particles}_{n_dims}_"
    fname_out += f"{step_size_importance}_{step_size_brute}_"
    fname_out += f"{fname_brute[6]}_{fname_brute[7]}_"
    fname_out += "std_mc_plot.png"

    #fig.savefig(fname = "../fig/" + fname_out, dpi=300)

    plt.show()


def plot_energy_MC(method, n_particles, n_dims, n_mc_cycles, step_size, numerical, interaction, alpha_place):

    input_energies = read_all_files(
        filter_method = method,
        filter_n_particles = n_particles,
        filter_n_dims = n_dims,
        filter_n_mc_cycles = n_mc_cycles,
        filter_step_size = step_size,
        filter_numerical = numerical,
        filter_interaction = interaction,
        filter_data_type = "energies")[0]

    input_particles = read_all_files(
        filter_method = method,
        filter_n_particles = n_particles,
        filter_n_dims = n_dims,
        filter_n_mc_cycles = n_mc_cycles,
        filter_step_size = step_size,
        filter_numerical = numerical,
        filter_interaction = interaction,
        filter_data_type = "particles")[0]

    alphas = input_energies.data[0, :]
    energies = input_energies.data[1:, alpha_place]

    exp_energy_ = input_particles.data[alpha_place, 2]

    mc_cycles = np.arange(n_mc_cycles)

    exp_energy = exp_energy_*np.ones(n_mc_cycles)

    fig = plt.figure(figsize=(9, 7))
    plt.grid()
    plt.plot(mc_cycles, energies, label = r"Brute-force, $\alpha=$"+f"{alphas[alpha_place]}")
    plt.plot(mc_cycles, exp_energy, linestyle="dashed", label="Expectation energy")
    plt.legend()
    #plt.xticks(mc_cycles)
    plt.xlabel(r"Number of MC-cycles")
    plt.ylabel(r"Local energy")
    plt.show()



def plot_acceptance_step_size(method, n_particles, n_dims, n_mc_cycles, step_sizes, numerical, interaction):

    avg_acceptance = []
    std_acceptance = []
    acceptance = []
    labels = []
    for step in step_sizes:

        input = read_all_files(
            filter_method = method,
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "particles")[0]

        avg = np.mean(input.data[:,4])
        std = np.std(input.data[:,4])
        avg_acceptance.append(avg)
        std_acceptance.append(std)
        acceptance.append(input.data[:,4])
        labels.append(r"$\alpha = $" + f"{step}")

    acceptance = np.array(acceptance)
    avg_acceptance = np.array(avg_acceptance)
    std_acceptance = np.array(std_acceptance)
    step_sizes = np.array(step_sizes)

    fig = plt.figure(figsize=(9, 7))
    plt.grid()
    plt.plot(step_sizes, avg_acceptance, "o", color="k")
    plt.xticks(step_sizes)
    plt.xlabel(r"step size")
    plt.ylabel("Acceptance rate")
    plt.ylim(0.4, 1.1)

    fname_out = f"{method}_{n_particles}_{n_dims}_acceptance_step_size.png"
    fig.tight_layout(pad=2)
    fig.savefig(fname = "../fig/" + fname_out, dpi=300)
    plt.show()


def plot_acceptance_MC(cycle_list, n_particles, n_dims, step_size_brute, step_size_importance, numerical, interaction):

    acceptance_brute = []
    acceptance_importance = []

    error_brute = []
    error_importance = []

    for n_mc_cycles_exponent in cycle_list:

        #print("MC: ", n_mc_cycles_exponent)
        n_mc_cycles = 2**n_mc_cycles_exponent

        brute = read_all_files(
            filter_method = "brute",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_brute,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "particles")[0]

        importance = read_all_files(
            filter_method = "importance",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_importance,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "particles")[0]

        acceptance_brute.append(np.mean(brute.data[:,4]))
        acceptance_importance.append(np.mean(importance.data[:,4]))

        error_brute.append(np.mean(brute.data[:,1]/n_mc_cycles))
        error_importance.append(np.mean(importance.data[:,1]/n_mc_cycles))

    acceptance_brute = np.array(acceptance_brute)
    acceptance_importance = np.array(acceptance_importance)

    error_brute = np.array(error_brute)
    error_importance = np. array(error_importance)

    mc_cycles = np.array(cycle_list)

    #fig = plt.figure(figsize=(9, 7))
    #plt.grid()

    #plt.plot(mc_cycles, acceptance_brute, color="tab:blue", label = r"A brute")
    #plt.plot(mc_cycles, acceptance_importance, color="tab:green", label=r"A importance")


    #plt.legend()
    #plt.xticks(mc_cycles)
    #plt.ylim(0.8, 1.05)
    #plt.xlabel(r"Number of MC-cycles")
    #plt.ylabel(r"acceptance")

    #fname_importance = importance.fname.split("_")
    #fname_brute = brute.fname.split("_")

    #fname_out = f"brute_importance_{n_particles}_{n_dims}_"
    #fname_out += f"{step_size_importance}_{step_size_brute}_"
    #fname_out += f"{fname_brute[6]}_{fname_brute[7]}_"
    #fname_out += "error_blocking_mc_plot.png"
    #fig.tight_layout(pad=2)
    #fig.savefig(fname = "../fig/" + fname_out, dpi=300)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False, figsize=(9,7), constrained_layout=True)
    labels = ["acceptance", "error"]
    ax[0].grid()
    ax[0].plot(mc_cycles, acceptance_brute, color="tab:blue", label="brute")
    ax[0].plot(mc_cycles, acceptance_importance, color="tab:green", label="importance")
    ax[0].set_ylabel("acceptance")

    ax[1].grid()
    ax[1].plot(mc_cycles, error_brute, color="tab:blue", label="brute")
    ax[1].plot(mc_cycles, error_importance, color="tab:green", label="importance")
    ax[1].set_ylabel("error")

    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines[:3], labels[:3], loc = 'lower right')
    plt.xlabel("Number of MC-cycles")
    plt.show()





if __name__ == '__main__':
    plot_acceptance_MC(cycle_list           =[8,10,12,14,16,18,20],
                       n_particles          = 10,
                       n_dims               = 3,
                       step_size_brute      = 0.2,
                       step_size_importance = 0.01,
                       numerical            = False,
                       interaction          = False,
                       )
    """
    plot_acceptance_step_size(method = "brute",
                              n_particles = 10,
                              n_dims      = 3,
                              n_mc_cycles = 2**20,
                              step_sizes  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                              numerical   = False,
                              interaction = False)
    """
    """
    met = "brute"
    step = 0.2
    plot_energy(method      = met,
                n_particles = 10,
                n_mc_cycles = 2**20,
                step_size   = step,
                numerical   = False,
                interaction = False
                )
    plot_energy(method      = met,
                n_particles = 100,
                n_mc_cycles = 2**20,
                step_size   = step,
                numerical   = False,
                interaction = False
                )
    plot_energy(method      = met,
                n_particles = 500,
                n_mc_cycles = 2**20,
                step_size   = step,
                numerical   = False,
                interaction = False
                )

    """
    """
    plot_variance_MC(cycle_list           =[8,10,12,14,16,18,20],
                     n_particles          = 10,
                     n_dims               = 3,
                     step_size_brute      = 0.2,
                     step_size_importance = 0.01,
                     numerical            = False,
                     interaction          = False,
                     alpha_place          = 5
                     )
    """

#    plot_energy_MC_old(cycle_list         =[8,10,12,14,16,18,20],
#                     n_particles          = 10,
#                     n_dims               = 3,
#                     step_size_brute      = 0.2,
#                     step_size_importance = 0.01,
#                     numerical            = False,
#                     interaction          = False,
#                     alpha_place          = 4
#                     )
    """
    plot_energy_MC(method = "brute",
                   n_particles = 10,
                   n_dims=3,
                   n_mc_cycles=2**20,
                   step_size=0.2,
                   numerical=False,
                   interaction=False,
                   alpha_place=4)
    """
