import multiprocessing
import time
import matplotlib.pyplot as plt
# import matplotlib as mpl
import numpy as np
from boltzmann_machine import ImportanceSampling
from blocking import block
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
    proc, n_hidden, scale, mc, iterations, lr = arg_list
    omega = 1
    sigma = np.sqrt(1/omega)
    
    q = ImportanceSampling(
        n_particles = 1,
        n_dims = 1,
        n_hidden = n_hidden,
        n_mc_cycles = mc,
        max_iterations = iterations,
        # learning_rate = 0.05,
        learning_rate = lr,
        sigma = sigma,
        interaction = False,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
        rng_seed = 1337
    )
    q.initial_state(
        loc_scale_all = (0, scale)
    )
    
    q.solve(
        verbose = True if proc == 0 else False,
        save_state = True,
        load_state = True
    )

    q.timing = time.time() - timing
    print(f"Process {proc} finished in {q.timing:.3f}s with parameters {arg_list[1:]}")

    return q

def find_threshold(
    energies: np.ndarray,
    exact: float,
    tol: float
) -> int:
    """
    Find the iteration where the local energy is 'close enough'.

    Parameters
    ----------
    energies:
        Array of energies as a function of GD iterations.

    exact:
        The exact energy.

    tol:
        The tolerance for deciding if the energy has been reached.

    Returns
    -------
    i:
        The index of the GD iteration where the goal is achieved, or the
        index where the value was closest to the goal.
    """    
    for i in range(len(energies)):
        if abs(energies[i] - exact) < tol:
            return i

    # return -1
    return np.argmin(np.abs(energies - exact))

def compare_multiple():
    fig, ax = plt.subplots()

    hidden_nodes = [1, 2, 3, 4, 5, 6]
    n_hidden_nodes = len(hidden_nodes)
    goal_indices = np.zeros(n_hidden_nodes, dtype=int)
    hidden_nodes_error = np.zeros(n_hidden_nodes)
    iterations = np.arange(1, 200+1, 1)
    args = []
    scale = 0.5

    for i in range(n_hidden_nodes):
        args.append([i, hidden_nodes[i], scale])

    pool = multiprocessing.Pool()
    results = pool.map(parallel, args)

    for i in range(n_hidden_nodes):
        goal_indices[i] = find_threshold(
            energies = results[i].energies,
            exact = 0.5,
            tol = 0.1
        )

    for i in range(n_hidden_nodes):
        data = results[i].energy_mc_iter[goal_indices[i], :]
        _, hidden_nodes_error[i], _, _, _ = block(data, verbose=False)

    for i in range(n_hidden_nodes):
        energy = results[i].energies[goal_indices[i]]
        energy_std = hidden_nodes_error[i]
        end_time = np.sum(results[i].times[:goal_indices[i]])
        print(f"\({hidden_nodes[i]}\) & \({energy:.2f} \pm {energy_std:5.1g}\) & \({iterations[goal_indices[i]]:3d}\) & \({end_time:8.1f}\)" + r" \\")
    
    # colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    # for i in reversed(range(n_hidden_nodes)):
    #     goal_index = goal_indices[i]
    #     iterations = np.arange(results[i].max_iterations)
    #     ax.plot(iterations[:goal_index], results[i].energies[:goal_index], label=r"n_hidden: " + f"{args[i][1]}", color=colors[i])
    #     ax.plot(iterations[goal_index-1], results[i].energies[goal_index-1], "o", color=colors[i])

    # ax.set_xlabel("Iterations")
    # ax.set_ylabel("Energy")
    # ax.legend()
    # plt.show()

def bar_plot_comparison():
    hidden_nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scales = [0.5, 1, 1.5]
    n_scales = len(scales)
    n_hidden_nodes = len(hidden_nodes)
    goal_indices = np.zeros((n_scales, n_hidden_nodes), dtype=int)
    hidden_nodes_error = np.zeros((n_scales, n_hidden_nodes))
    res_array = np.empty((n_scales, n_hidden_nodes), dtype=object)
    end_iterations = np.zeros((n_scales, n_hidden_nodes))
    iterations = np.arange(1, 200+1, 1)
    args = []

    proc = 0
    for i in range(n_scales):
        for j in range(n_hidden_nodes):
            args.append([proc, hidden_nodes[j], scales[i]])
            proc += 1

    pool = multiprocessing.Pool()
    results = pool.map(parallel, args)

    proc = 0
    for i in range(n_scales):
        for j in range(n_hidden_nodes):
            res_array[i, j] = results[proc]
            goal_indices[i, j] = find_threshold(
                energies = results[proc].energies,
                exact = 0.5,
                tol = 0.1
            )
            proc += 1

    proc = 0
    for i in range(n_scales):
        for j in range(n_hidden_nodes):
            data = results[proc].energy_mc_iter[goal_indices[i, j], :]
            _, hidden_nodes_error[i, j], _, _, _ = block(data, verbose=False)
            proc += 1

    for i in range(n_scales):
        for j in range(n_hidden_nodes):
            end_iterations[i, j] = iterations[goal_indices[i, j]]

    x = np.arange(n_hidden_nodes)  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, end_iterations[0, :], width, label=r"$\sigma_{init} = $" + f"{res_array[0, 0].loc_scale_all[1]}")
    rects2 = ax.bar(x, end_iterations[1, :], width, label=r"$\sigma_{init} = $" + f"{res_array[1, 0].loc_scale_all[1]}")
    rects2 = ax.bar(x + width, end_iterations[2, :], width, label=r"$\sigma_{init} = $" + f"{res_array[2, 0].loc_scale_all[1]}")

    ax.set_ylabel("GD iterations")
    ax.set_xlabel("# hidden nodes")
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_nodes)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

def look_at_one():
    proc = 1
    hidden = [6]
    n_hidden = len(hidden)
    scale = 1.5
    # args = [[proc, hidden, scale]]
    args = []

    for i in range(n_hidden):
        args.append([i, hidden[i], scale])
    pool = multiprocessing.Pool()
    results = pool.map(parallel, args)

    plt.plot(range(results[0].max_iterations), results[0].energies)
    plt.hlines(y=0.5, xmin=0, xmax=100, linestyle="dashed", color="black")
    plt.show()

def look_at_more():
    proc = 1
    hidden = [2]
    n_hidden = len(hidden)
    scale = 1.5
    # scale = 1
    args = [
        [0, hidden[0], scale, int(2**16), 100, {"factor": 0.05, "init": 0.18}],
        [1, hidden[0], scale, int(2**18), 50, {"factor": 0.05, "init": 0.18}],
        [2, hidden[0], scale, int(2**20), 50, {"factor": 0.05, "init": 0.18}],
        [0, hidden[0], scale, int(2**16), 400, 0.01],
        [3, hidden[0], scale, int(2**16), 400, 0.02],
        [4, hidden[0], scale, int(2**16), 400, 0.005],
        [5, hidden[0], scale, int(2**16), 400, 0.001],
    ]
    # args = []

    # for i in range(n_hidden):
    #     args.append([i, hidden[i], scale])
    pool = multiprocessing.Pool()
    results = pool.map(parallel, args)

    for i in range(len(results)):
        plt.plot(range(results[i].max_iterations), results[i].energies, label=f"{results[i].n_mc_cycles}")
    plt.hlines(y=0.5, xmin=0, xmax=100, linestyle="dashed", color="black")
    plt.legend()
    plt.title(r"$\sigma_{init}$ = " + f"{results[0].loc_scale_all[1]}")
    plt.xlabel("Iterations")
    plt.ylabel(r"$E_L$")
    plt.show()

if __name__ == "__main__":
    # compare_multiple()
    # bar_plot_comparison()
    # look_at_one()
    look_at_more()
    pass