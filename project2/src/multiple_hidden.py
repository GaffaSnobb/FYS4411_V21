import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from boltzmann_machine import ImportanceSampling, BruteForce
import mpl_rcparams


def parallel(arg_list: list):
    """
    This function is meant to be run in parallel with
    multiprocessing.pool.

    Parameters
    ----------
    arg_list:
        A list of arguments to pass to the class constructor.
    """

    timing = time.time()
    proc, learning_rate, n_hidden, n_dims, n_particles, n_mc_cycles = arg_list
    omega = 1
    sigma = np.sqrt(1/omega)

    q = BruteForce(
        n_particles = n_particles,
        n_dims = n_dims,
        n_hidden = n_hidden,
        n_mc_cycles = n_mc_cycles,
        max_iterations = 100,
        learning_rate = learning_rate,
        sigma = sigma,
        interaction = False,
        omega = omega,
        brute_force_step_size = 1,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
        rng_seed = 1337
    )

    q.initial_state(
        loc_scale_all = (0, 1)
    )

    q.solve(verbose=False)

    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q


def process(learning_rate, hidden, n_mc=int(2**18), n_dims = 1, n_particles=1):
    """
    Process in paralell for hidden nodes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    and print results as latex table for the report.

    Parameters
    ----------

    learning_rate:
        The learning rate desired, can either be a float or a dictionary on the
        form {"factor": float, "init": float})

    hidden:
        List of numbers of hidden nodes (integers)

    n_mc:
        The number of Monte Carlo cycles, must be an integer on the form 2^N

    n_dims:
        The number of dimensions of the system.

    n_particles:
        The number of particles
    """

    # Set up list of list of arguments
    args = []
    for i in range(len(hidden)):
        args.append([i, learning_rate, hidden[i], n_dims, n_particles, n_mc])

    # This is where the magic happens
    t0 = time.time()
    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)
    t1 = time.time()

    print()
    print("Time elapsed: ", t1-t0)
    print()


    # Print latex compatible table of the results
    for i in range(len(res)):
        n = res[i].n_hidden
        e = res[i].energies[-1]
        b = res[i].blocking_final[-1]
        err = np.abs((info[0]-e))/info[0]
        end_time = np.sum(res[i].times)

        s = f"\( {n} \) & \( {e:.5f} \pm {b:5.1g} \) & \( {err*100:.5f}\) & \( {end_time:.2f} \)"+r" \\"
        print(s)


def energy_iteration_plot(learning_rate, hidden, max_iterations = 100,  variety = "constant", info=""):
    """
    Read files and plot energy vs iteration results

    Parameters
    ----------

    learning_rate:
        The learning rate to plot for, must be a float.

    hidden:
        List of hidden nodes to plot for

    max_iterations:
        The maximum number of gradient descent iterations to use

    variety:
        If you want to plot for constant or variable learning rate, options are
        "constant" or "variable"

    info:
        Additional string to add to the figure name

    """

    fig, ax = plt.subplots()

    path = "tmp/multiple_hidden/"

    for i, node in enumerate(hidden):

        if variety == "constant":
            folder = f"brute_1_1_{node}_1_262144_100_{learning_rate}_1.0_False_1_all(0,1)_1337_1/"

        elif variety == "variable":
            folder = f"brute_1_1_{node}_1_262144_100_(0.05,{learning_rate})_1.0_False_1_all(0,1)_1337_1/"

        energy = np.load(path+folder+"energies.npy")

        ax.plot(range(max_iterations), energy[:max_iterations], label=r"Hidden, $N_b$ = "+f"{node}")

    ax.hlines(y=0.5, xmin=0, xmax=max_iterations, label=r"Exact ($E_L = 0.5$ a.u.)", linestyle="dashed", color="black")

    ax.set_xlabel("GD iterations")
    ax.set_ylabel(r"$E_L$ [a.u.]")
    ax.legend(fontsize=12)

    fig.savefig(fname=f"../fig/multiple_hidden/energy_iterations_{variety}_{learning_rate}_{max_iterations}_{info}.png", dpi=300)
    plt.close()



def energy_acceptance_subplots(learning_rates, hidden, max_iterations = 100,  variety = "constant", info=""):
    """
    Read files and plot energy vs iteration results

    Parameters
    ----------

    learning_rates:
        List of the learning rate to plot for

    hidden:
        List of hidden nodes to plot for

    max_iterations:
        The maximum number of gradient descent iterations to use

    variety:
        If you want to plot for constant or variable learning rate, options are
        "constant" or "variable"

    info:
        Additional string to add to the figure name

    """

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False)    #plt.subplots()

    path = "tmp/multiple_hidden/"

    for learning_rate in learning_rates:

        for i, node in enumerate(hidden):

            if variety == "constant":
                folder = f"brute_1_1_{node}_1_262144_100_{learning_rate}_1.0_False_1_all(0,1)_1337_1/"
                lr_str = r"$\eta=$"+f"{learning_rate:.2f}"

            elif variety == "variable":
                folder = f"brute_1_1_{node}_1_262144_100_(0.05,{learning_rate})_1.0_False_1_all(0,1)_1337_1/"
                lr_str = r"$\eta_{init}=$"+f"{learning_rate:.2f}, "+ r"$f_{\eta}=0.05$"

            energy = np.load(path+folder+"energies.npy")
            rate = np.load(path+folder+"acceptance_rates.npy")

            ax[0].plot(range(max_iterations), energy[:max_iterations], label=r"Hidden, $N_b$ = "+f"{node}, "+lr_str)
            ax[1].plot(range(max_iterations), rate[:max_iterations], label=r"Hidden, $N_b$ = "+f"{node}"+lr_str)

        ax[0].hlines(y=0.5, xmin=0, xmax=max_iterations, label=r"Exact ($E_L = 0.5$ a.u.)", linestyle="dashed", color="black")

        ax[1].set_xlabel("GD iterations")
        ax[0].set_ylabel(r"$E_L$ [a.u.]")
        ax[1].set_ylabel(r"Acceptance rate")

        ax[0].legend(fontsize=12)

    fig.savefig(fname=f"../fig/multiple_hidden/energy_acceptance_{variety}_{learning_rate}_{max_iterations}_{info}.png", dpi=300)
    plt.close()



def time_nodes_plot(learning_rates, hidden):
    """
    Make plot of time versus nodes

    Parameters
    ----------

    learning_rate:
        The learning rate to plot for, must be a float.

    hidden:
        List of hidden nodes to plot for
    """

    fig, ax = plt.subplots()

    path = "tmp/multiple_hidden/"

    times_c = np.zeros((len(learning_rates), len(hidden)))
    times_v = np.zeros((len(learning_rates), len(hidden)))

    colours = ["tab:green", "tab:blue", "tab:red"]

    for i, lr in enumerate(learning_rates):
        for j, node in enumerate(hidden):

            fname_c = f"brute_1_1_{node}_1_262144_100_{lr}_1.0_False_1_all(0,1)_1337_1/times.npy"
            fname_v = f"brute_1_1_{node}_1_262144_100_(0.05,{lr})_1.0_False_1_all(0,1)_1337_1/times.npy"

            label_string_c = r"$\eta=$"+f"{lr:.2f}"
            label_string_v = r"$\eta_{init}=$"+f"{lr:.2f}, "+ r"$f_{\eta}=0.05$"

            data_c = np.load(path+fname_c)
            data_v = np.load(path+fname_v)

            times_c[i, j] = np.sum(data_c)
            times_v[i, j] = np.sum(data_v)

        ax.plot(hidden, times_c[i], color=colours[i], linewidth=1, label=label_string_c)
        ax.plot(hidden, times_v[i], color=colours[i], linewidth=1, linestyle="dashed", label=label_string_v)

        ax.plot(hidden, times_c[i], ".", color=colours[i])
        ax.plot(hidden, times_v[i], ".", color=colours[i])

    ax.set_xlabel("Hidden nodes")
    ax.set_ylabel("Time [s]")

    ax.set_xticks(np.arange(hidden[0], hidden[-1]+1, 2))

    ax.legend(fontsize=12)

    fig.savefig(fname=f"../fig/multiple_hidden/time_nodes.png", dpi=300)
    plt.close()


def energy_acceptance_compare_subplots(learning_rate, hidden, max_iterations = 100, info=""):
    """
    Read files and plot energy vs iteration results

    Parameters
    ----------

    learning_rates:
        learning rate to plot for

    hidden:
        List of hidden nodes to plot for

    max_iterations:
        The maximum number of gradient descent iterations to use

    info:
        Additional string to add to the figure name

    """

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex="col", sharey=False)    #plt.subplots()

    path = "tmp/multiple_hidden/"

    folder_c = f"brute_1_1_{hidden}_1_262144_100_{learning_rate}_1.0_False_1_all(0,1)_1337_1/"
    folder_v = f"brute_1_1_{hidden}_1_262144_100_(0.05,{learning_rate})_1.0_False_1_all(0,1)_1337_1/"

    lr_str_c = r"$\eta=$"+f"{learning_rate:.2f}"
    lr_str_v = r"$\eta_{init}=$"+f"{learning_rate:.2f}, "+ r"$f_{\eta}=0.05$"

    energy_c = np.load(path+folder_c+"energies.npy")
    rate_c = np.load(path+folder_c+"acceptance_rates.npy")

    energy_v = np.load(path+folder_v+"energies.npy")
    rate_v = np.load(path+folder_v+"acceptance_rates.npy")

    ax[0].plot(range(max_iterations), energy_c[:max_iterations], label=r"Hidden, $N_b$ = "+f"{hidden}, "+lr_str_c)
    ax[0].plot(range(max_iterations), energy_v[:max_iterations], label=r"Hidden, $N_b$ = "+f"{hidden}, "+lr_str_v)

    ax[1].plot(range(max_iterations), rate_c[:max_iterations], label=r"Hidden, $N_b$ = "+f"{hidden}"+lr_str_c)
    ax[1].plot(range(max_iterations), rate_v[:max_iterations], label=r"Hidden, $N_b$ = "+f"{hidden}"+lr_str_v)

    ax[0].hlines(y=0.5, xmin=0, xmax=max_iterations, label=r"Exact ($E_L = 0.5$ a.u.)", linestyle="dashed", color="black")

    ax[1].set_xlabel("GD iterations")
    ax[0].set_ylabel(r"$E_L$ [a.u.]")
    ax[1].set_ylabel(r"Acceptance rate")

    ax[0].legend(fontsize=10)

    fig.savefig(fname=f"../fig/multiple_hidden/compare_energy_acceptance_{learning_rate}_{max_iterations}_{info}.png", dpi=300)
    plt.close()


def compare_the_two_best():

    path = "tmp/multiple_hidden/"

    folder_c = f"brute_1_1_2_1_262144_100_0.08_1.0_False_1_all(0,1)_1337_1/"
    folder_v = f"brute_1_1_20_1_262144_100_(0.05,0.1)_1.0_False_1_all(0,1)_1337_1/"

    energy_c = np.load(path+folder_c+"energies.npy")
    energy_v = np.load(path+folder_v+"energies.npy")

    time_c = np.load(path+folder_c+"times.npy")
    time_v = np.load(path+folder_v+"times.npy")

    block_c = np.load(path+folder_c+"blocking_all.npy")
    block_v = np.load(path+folder_v+"blocking_all.npy")

    relerr_c = np.abs(energy_c - 0.5)/0.5
    relerr_v = np.abs(energy_v - 0.5)/0.5

    min_c = relerr_c.argmin()
    min_v = relerr_v.argmin()

    print(f"constant: E = {energy_c[min_c]:.5f} \pm {block_c[min_c]:5.1g},  Rel.err = {relerr_c[min_c]:5.1g}, GD = {min_c}, time = {np.sum(time_c[:min_c]):.2f}")
    print(f"variable: E = {energy_v[min_v]:.5f} \pm {block_v[min_v]:5.1g},  Rel.err = {relerr_v[min_v]:5.1g}, GD = {min_v}, time = {np.sum(time_v[:min_v]):.2f}")

    """
    n = 20

    fig, ax = plt.subplots()
    ax.plot(range(n, max_iterations), relerr_c[n:max_iterations]*100, label=r"$\eta=0.08$")
    ax.plot(range(n, max_iterations), relerr_v[n:max_iterations]*100, label=r"$\eta_{init}=0.1$, $f_{\eta}=0.05$")
    ax.set_xlabel("GD iterations")
    ax.set_ylabel(r"Relative error [%]")
    ax.legend(fontsize=12)
    plt.show()
    """

def run_processes():
    """
    Run all necessary processes
    """
    hidden = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]

    process(learning_rate = 0.1, hidden = hidden)
    process(learning_rate = 0.08, hidden = hidden)
    process(learning_rate = 0.01, hidden = hidden)

    process(learning_rate = {"factor": 0.05, "init": 0.1}, hidden = hidden)
    process(learning_rate = {"factor": 0.05, "init": 0.08}, hidden = hidden)
    process(learning_rate = {"factor": 0.05, "init": 0.01}, hidden = hidden)


def make_plots():
    """
    Only works when you run for 2^18 mc cycles, 1p in 1d
    """

    hidden = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    hidden_short = [4, 8, 10, 20]


    energy_iteration_plot(
        learning_rate = 0.01,
        hidden = [4, 8, 10, 20],
        max_iterations = 100,
        variety = "variable",
        info = ""
        )

    # plot processing time as a function of hidden nodes
    time_nodes_plot(
        learning_rates = [0.1, 0.08, 0.01],
        hidden = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
        )

    energy_acceptance_compare_subplots(
        learning_rate = 0.1,
        hidden = 20,
        max_iterations = 100,
        info = ""
        )



if __name__ == "__main__":
    #run_processes()
    make_plots()
    compare_the_two_best()
    #testing()
