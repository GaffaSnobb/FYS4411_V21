from typing import Union
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from boltzmann_machine import ImportanceSampling
from blocking import block

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
    proc, n_hidden = arg_list
    omega = 1
    sigma = np.sqrt(1/omega)
    
    q = ImportanceSampling(
        n_particles = 1,
        n_dims = 1,
        n_hidden = n_hidden,
        n_mc_cycles = int(2**12),
        max_iterations = 100,
        learning_rate = 0.05,
        sigma = sigma,
        interaction = False,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
        # rng_seed = 1337
    )
    q.initial_state(
        loc_scale_all = (0, 1)
    )
    q.solve(verbose=False, save_state=True)

    q.timing = time.time() - timing
    print(f"Process {proc} finished in {q.timing:.3f}s with parameters {arg_list[1:]}")

    return q

def find_threshold(
    energies: np.ndarray,
    exact: float,
    tol: float
) -> Union[int, None]:
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
    idx:
        The index of the GD iteration where the goal is achieved, or
        None if goal is not achieved.
    """
    idx = None
    
    for i in range(len(energies)):
        if abs(energies[i] - exact) < tol:
            idx = i
            break

    return idx

def main():
    fig, ax = plt.subplots()

    hidden_nodes = [1, 2, 3, 4, 5, 6]#, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    n_hidden_nodes = len(hidden_nodes)
    goal_indices = np.zeros(n_hidden_nodes)
    hidden_nodes_error = np.zeros(n_hidden_nodes)
    args = []

    for i in range(n_hidden_nodes):
        args.append([i, hidden_nodes[i]])

    pool = multiprocessing.Pool()
    results = pool.map(parallel, args)

    for i in range(n_hidden_nodes):
        goal_indices[i] = find_threshold(
            energies = results[i].energies,
            exact = 0.5,
            tol = 0.1
        )
    
    print(goal_indices)

    # for i in range(n_hidden_nodes):
    #     data = results[i].energy_mc_iter[-1, :]
    #     _, hidden_nodes_error[i], _, _, _ = block(data, verbose=False)

    # for i in range(n_hidden_nodes):
    #     print(f"{hidden_nodes[i]} \pm {hidden_nodes_error[i]} & ")
    
    # for i in range(len(results)):
    #     ax.plot(range(results[i].max_iterations), results[i].energies, label=r"n_hidden: " + f"{args[i][1]}")

    # ax.set_xlabel("Iterations")
    # ax.set_ylabel("Energy")
    # ax.legend()
    # plt.show()

if __name__ == "__main__":
    main()