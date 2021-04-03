import numpy as np
import matplotlib.pyplot as plt
from read_from_file import read_all_files
import os
np.seterr(divide='ignore', invalid='ignore')
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:green", "tab:purple","tab:red", "tab:orange", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"])
mpl.rcParams.update({'font.size': 15})

def block(x, verbose=True):
    """
    Credit: Marius Jonsson
    Jonsson, M. (2018). Standard error estimation by an automated blocking method. Physical Review E, 98(4), 043304.
    """
    # preliminaries
    n = len(x)
    d = int(np.log2(n))
    s, gamma, error_array = np.zeros(d), np.zeros(d), np.zeros(d)
    mu = np.mean(x)

    # Calculate the autocovariance and variance for the data
    gamma[0] = (n)**(-1)*np.sum( (x[0:(n-1)] - mu) * (x[1:n] - mu) )
    s[0] = np.var(x)
    error_array[0] = (s[0]/n)**.5

    # estimate the auto-covariance and variances for each blocking transformation
    for i in np.arange(1, d):

        # perform blocking transformation
        x = 0.5*(x[0::2] + x[1::2])

        n = len(x)

        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*np.sum( (x[0:(n-1)] - mu) * (x[1:n] - mu) )

        # estimate variance of x
        s[i] = np.var(x)

        # estimate the error
        error_array[i] = (s[i]/n)**.5


    # generate the test observator M_k from the theorem
    M = (np.cumsum( ((gamma/s)**2*2**np.arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    # alpha=0.01
    """
    q = np.array([6.634897,9.210340, 11.344867, 13.276704, 15.086272, 16.811894,
                18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967,
                27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306,
                36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820,
                44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])
    """
    # alpha= 0.05
    q = np.array([3.841, 5.991, 7.815, 9.488, 11.070, 12.592, 14.067, 15.507,
                 16.919, 18.307, 19.675, 21.026, 22.362, 23.685, 24.996, 26.296,
                 27.587, 28.869, 30.144, 31.410, 32.671, 33.924, 35.172, 36.415,
                 37.652, 38.885, 40.113, 41.337, 42.557, 43.773, 44.985, 46.194,
                 47.400, 48.602, 49.802, 50.998, 52.192, 53.384, 54.572, 55.758,
                 56.942, 58.124, 59.304, 60.481, 61.656, 62.830, 64.001, 65.171,
                 66.339, 67.505, 68.669, 69.832, 70.993, 72.153, 73.311, 74.468,
                 75.624, 76.778, 77.931, 79.082, 80.232, 81.381, 82.529, 83.675,
                 84.821, 85.965, 87.108, 88.250, 89.391, 90.531, 91.670, 92.808,
                 93.945, 95.081, 96.217, 97.351, 98.484, 99.617, 100.749, 101.879,
                 103.010, 104.139, 105.267, 106.395, 107.522, 108.648, 109.773,
                 110.898, 112.022, 113.145, 114.268, 115.390, 116.511, 117.632,
                 118.752, 119.871, 120.990, 122.108, 123.225, 124.342, 124.342])

    # use magic to determine when we should have stopped blocking
    for k in np.arange(0, d):
        if(M[k] < q[k]):
            break
    if (k >= d-1):
        if verbose:
            print ("Warning: Use more data")

    best_error = error_array[k]
    original_error = error_array[0]

    if verbose:
        print(f"avg: {mu:.6f}, error(orig): {original_error:.6f}, error(block): {best_error:.6f}, iterations: {k}\n")

    return mu, best_error, original_error, k, error_array

def create_folder(path):
    """
    creates a folder
    ---------------
        path: str, the path you want to create
    ---------------
    """

    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)


def blocking_analysis(n_particles, n_dims, mc_cycles, method, numerical=False, interaction=False):
    """
    Do the blocking analysis on one of the datasets named output_*_energy_.txt
    generated using the main script.
    ---------------
        n_particles:    int, number of particles
        n_dims:         int, number of dimensions
        mc_cycles:      int, number of monte carlo cycles
        method:         str, options are "brute", "importance" or "gradient"
        numerical:      bool, if numerical differentiation is used
        interaction:    bool, if there are interaction between particles
    ---------------
    """

    # Create folder for the figures to be stored

    num = "analytic" if not numerical else "numerical"
    inter = "nointeraction" if not interaction else "interaction"
    path = f"../fig/blocking_{method}_{n_particles}_{n_dims}_{num}_{inter}/"

    create_folder(path)


    print(45*"_"+"\n", method, 45*"_"+"\n", sep="\n")

    input_energies = read_all_files(
        filter_method = method,
        filter_n_particles = n_particles,
        filter_n_dims = n_dims,
        filter_n_mc_cycles = mc_cycles,
        filter_step_size = None,
        filter_numerical = numerical,
        filter_interaction = interaction,
        filter_data_type = "energies"
    )

    input_particles = read_all_files(
        filter_method = method,
        filter_n_particles = n_particles,
        filter_n_dims = n_dims,
        filter_n_mc_cycles = mc_cycles,
        filter_step_size = None,
        filter_numerical = numerical,
        filter_interaction = interaction,
        filter_data_type = "particles"
    )

    print(input_energies[0].fname)
    print(input_energies[0].fname)
    print()

    # Get the array of alpha values
    alphas = input_energies[0].data[0,:]

    #header = "alpha\tenergy\t\tblocking_error\toriginal_error\tCPU\tacceptance"
    header = "alpha\tenergy\t\tblocking_error\tCPU"

    print(header)

    for i in range(len(alphas)):
        # Loop over alpha values and do blocking to find error

        data = input_energies[0].data[1:,i]
        #energy, blocking_error, original_error, iterations, error_array = block(data, verbose=True)
        energy, blocking_error, original_error, iterations, error_array = block(data, verbose=False)
#        s = f"{alphas[i]:.1f}\t{energy:.6f}\t{blocking_error:.6f}\t{original_error:.6f}\t{input_particles[0].data[i, 3]}\t{input_particles[0].data[i, 4]}"

        s = f"{alphas[i]:.1f}\t{energy:.4f}\t{blocking_error:.4f}\t{input_particles[0].data[i, 3]:.2f}"
        print(s)

        iter = np.arange(len(error_array))
        block_line = np.ones(len(error_array)) * blocking_error
        orig_line =  np.ones(len(error_array)) * original_error

        fig = plt.figure()
        plt.grid()
        plt.plot(iter, error_array, ".", color="k", label=r"$Error, \alpha=$"+f"{alphas[i]}")
        plt.plot(iter, block_line, linestyle="dashed", color="tab:blue", label="Optimal")

        plt.xticks(np.arange(min(iter), max(iter)+1, 1.0))
        plt.xlabel("Blocking iterations, k")
        plt.ylabel(r"Sample Error, $\sqrt{\sigma^2_k \ / \ n_k}$")
        plt.legend()
        fig.savefig(f"{path}/alpha_{alphas[i]}.png")

def plot_variance_MC(cycle_list, n_particles, n_dims, step_size_brute, step_size_importance, numerical, interaction, alpha_place):

    variances_brute = []
    variances_importance = []

    blocking_brute = []
    blocking_importance = []

    for n_mc_cycles_exponent in cycle_list:

        print("MC: ", n_mc_cycles_exponent)
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

        brute_energies = read_all_files(
            filter_method = "brute",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_brute,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "energies")[0]

        importance = read_all_files(
            filter_method = "importance",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_importance,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "particles")[0]

        importance_energies = read_all_files(
            filter_method = "importance",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_importance,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "energies")[0]

        variances_brute.append((brute.data[:, 1][alpha_place]/n_mc_cycles)**0.5)
        variances_importance.append((importance.data[:, 1][alpha_place]/n_mc_cycles)**0.5)

        data_brute = brute_energies.data[1:, alpha_place]
        data_importance = importance_energies.data[1:, alpha_place]

        energy, blocking_error_brute, original_error, iterations, error_array = block(data_brute, verbose=True)
        energy, blocking_error_importance, original_error_importance, iterations, error_array = block(data_importance, verbose=True)

        blocking_brute.append(blocking_error_brute)
        blocking_importance.append(blocking_error_importance)

        alpha = brute.data[:,0][alpha_place]

    variances_brute = np.array(variances_brute)
    variances_importance = np.array(variances_importance)

    blocking_brute = np.array(blocking_brute)
    blocking_importance = np.array(blocking_importance)


    mc_cycles = np.array(cycle_list)

    fig = plt.figure(figsize=(9, 7))
    plt.grid()
    plt.plot(mc_cycles, variances_brute, color = "tab:blue",  linestyle="dashed", label = "Brute-force original error")
    plt.plot(mc_cycles, variances_importance, color = "tab:green",  linestyle="dashed", label= "Importance original error")
    plt.plot(mc_cycles, blocking_brute, color="tab:blue", label = "Brute-force blocking error")
    plt.plot(mc_cycles, blocking_importance, color="tab:green", label= "Importance blocking error")

    plt.legend()
    plt.xticks(mc_cycles)
    plt.xlabel(r"Number of MC-cycles")
    plt.ylabel(r"Error,  $\sigma(m)$")

    fname_importance = importance.fname.split("_")
    fname_brute = brute.fname.split("_")

    fname_out = f"brute_importance_{n_particles}_{n_dims}_"
    fname_out += f"{step_size_importance}_{step_size_brute}_"
    fname_out += f"{fname_brute[6]}_{fname_brute[7]}_"
    fname_out += "error_mc_plot.png"
    fig.tight_layout(pad=2)
    fig.savefig(fname = "../fig/" + fname_out, dpi=300)

    plt.show()







def plot_error_MC(cycle_list, n_particles, n_dims, step_size_brute, step_size_importance, numerical, interaction, alpha_place):

    error_brute = []
    error_importance = []

    original_brute = []
    original_importance = []

    for n_mc_cycles_exponent in cycle_list:

        print("MC: ", n_mc_cycles_exponent)
        n_mc_cycles = 2**n_mc_cycles_exponent

        brute = read_all_files(
            filter_method = "brute",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_brute,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "energies")[0]

        importance = read_all_files(
            filter_method = "importance",
            filter_n_particles = n_particles,
            filter_n_dims = n_dims,
            filter_n_mc_cycles = n_mc_cycles,
            filter_step_size = step_size_importance,
            filter_numerical = numerical,
            filter_interaction = interaction,
            filter_data_type = "energies")[0]

        alphas = importance.data[0,:]

        tmp_brute = np.zeros(len(alphas))
        tmp_importance = np.zeros(len(alphas))

        tmp_brute_original = np.zeros(len(alphas))
        tmp_importance_original = np.zeros(len(alphas))

        for i in range(len(alphas)):

            data_brute = brute.data[1:,i]
            data_importance = importance.data[1:,i]

            blocking_error_brute, original_error_brute = block(data_brute, verbose=False)[1:3]
            blocking_error_importance, original_error_importance = block(data_importance, verbose=False)[1:3]

            tmp_brute[i] = blocking_error_brute
            tmp_importance[i] = blocking_error_importance

            tmp_brute_original[i] = original_error_brute
            tmp_importance_original[i] = original_error_importance


        error_brute.append(np.mean(tmp_brute))
        error_importance.append(np.mean(tmp_importance))

        original_brute.append(np.mean(tmp_brute_original))
        original_importance.append(np.mean(tmp_importance_original))


    error_brute = np.array(error_brute)
    error_importance = np.array(error_importance)

    original_brute = np.array(original_brute)
    original_importance = np.array(original_importance)

    mc_cycles = np.array(cycle_list)

    fig = plt.figure(figsize=(9, 7))
    plt.grid()

    plt.plot(mc_cycles, error_brute, color="tab:blue", label = r"$\sigma_b$ brute-force")
    plt.plot(mc_cycles, original_brute, color="tab:blue", linestyle="dashed", label=r"$\sigma$, brute-force")
    plt.plot(mc_cycles, error_importance, color="tab:green", label=r"$\sigma_b$, importance")
    plt.plot(mc_cycles, original_importance, color="tab:green", linestyle="dashed", label=r"$\sigma$, importance")

    plt.legend()
    plt.xticks(mc_cycles)
    plt.xlabel(r"Number of MC-cycles")
    plt.ylabel(r"Error,  $\sigma(m)$")

    fname_importance = importance.fname.split("_")
    fname_brute = brute.fname.split("_")

    fname_out = f"brute_importance_{n_particles}_{n_dims}_"
    fname_out += f"{step_size_importance}_{step_size_brute}_"
    fname_out += f"{fname_brute[6]}_{fname_brute[7]}_"
    fname_out += "error_blocking_mc_plot.png"
    fig.tight_layout(pad=2)
    fig.savefig(fname = "../fig/" + fname_out, dpi=300)

    plt.show()






def main():
    """
    blocking_analysis(method = "importance",
                      n_particles = 500,
                      n_dims = 3,
                      mc_cycles= int(2**20),
                      numerical=True,
                      interaction=False)
    """

    plot_error_MC(cycle_list = [6,8,10,12,14,16,18,20],
                      n_particles=10,
                      n_dims=3,
                      step_size_brute=0.2,
                      step_size_importance=0.01,
                      numerical=False,
                      interaction=False,
                      alpha_place=5
                      )

if __name__ == '__main__':
    main()
