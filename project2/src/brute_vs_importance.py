import multiprocessing
import time
import numpy as np
from boltzmann_machine import ImportanceSampling, BruteForce

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
    omega = 1
    sigma = np.sqrt(1/omega)

    proc, method, n_mc_cycles = arg_list
    n_particles = 1
    n_dims = 1
    n_hidden = 2
    max_iterations = 50
    learning_rate = 0.05
    interaction = True
    rng_seed = 1337
    scale = 2
    parent_data_directory = (__file__.split(".")[0]).split("/")[-1]
    
    if method == "importance":
        q = ImportanceSampling(
            n_particles = n_particles,
            n_dims = n_dims,
            n_hidden = n_hidden,
            n_mc_cycles = n_mc_cycles,
            max_iterations = max_iterations,
            learning_rate = learning_rate,
            sigma = sigma,
            interaction = interaction,
            omega = omega,
            diffusion_coeff = 0.5,
            time_step = 0.05,
            rng_seed = rng_seed,
            parent_data_directory = parent_data_directory,
        )

    elif method == "brute":
        q = BruteForce(
            n_particles = n_particles,
            n_dims = n_dims,
            n_hidden = n_hidden,
            n_mc_cycles = n_mc_cycles,
            max_iterations = max_iterations,
            learning_rate = learning_rate,
            sigma = sigma,
            interaction = interaction,
            omega = omega,
            brute_force_step_size = 1,
            rng_seed = rng_seed,
            parent_data_directory = parent_data_directory,
        )
    q.initial_state(
        loc_scale_all = (0, scale)
    )
    q.solve(
        verbose = True if proc == 0 else False,
        save_state = True,
        load_state = True,
        calculate_blocking_all = True
    )
    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")
    
    return q

def main():
    """
    Produce comparison data of brute force and importance sampling.
    Print the results in a LaTeX formatted table.
    """
    args = [
        ["brute", int(2**18)],
        ["importance", int(2**18)],
        ["brute", int(2**17)],
        ["importance", int(2**17)],
        ["brute", int(2**16)],
        ["importance", int(2**16)],
        ["brute", int(2**15)],
        ["importance", int(2**15)],
        ["brute", int(2**14)],
        ["importance", int(2**14)],
        ["brute", int(2**13)],
        ["importance", int(2**13)],
        ["brute", int(2**12)],
        ["importance", int(2**12)],
        ["brute", int(2**11)],
        ["importance", int(2**11)],
        ["brute", int(2**10)],
        ["importance", int(2**10)],
    ]

    args = [[proc, method, mc] for proc, [method, mc] in enumerate(args)]
    exact = 0.5 # Exact energy of the system [a.u.]
    tol = 0.01  # Tolerance where the result is good enough [a.u.]
    pool = multiprocessing.Pool()
    results = pool.map(parallel, args)
    n_results = len(results)
    goal_indices = np.zeros(n_results, dtype=int)

    for i in range(n_results):
        goal_indices[i] = np.where(np.abs(results[i].energies - exact) < tol)[0][0]

    for i in range(n_results):
        """
        Print results from the iteration where the tolerance is
        achieved.
        """
        method = results[i].__str__().split()[0]
        n_mc_cycles = results[i].n_mc_cycles
        energy = results[i].energies[goal_indices[i]]
        energy_std = results[i].blocking_all[goal_indices[i]]
        timing = np.sum(results[i].times[:goal_indices[i]])
        gd_iterations_end = goal_indices[i]
        print(f"{method:11s} & {n_mc_cycles:6d} & \({energy:7.4f} \pm {energy_std:7.1g}\) & {(np.abs(energy - exact))*100/exact:7.4f} & {gd_iterations_end} & {timing:.1f}" + r" \\")

    print()
    for i in range(n_results):
        """
        Print results from the final iteration.
        """
        method = results[i].__str__().split()[0]
        n_mc_cycles = results[i].n_mc_cycles
        energy = results[i].energies[-1]
        energy_std = results[i].blocking_all[-1]
        timing = np.sum(results[i].times[:-1])
        gd_iterations_end = results[i].max_iterations
        print(f"{method:11s} & {n_mc_cycles:6d} & \({energy:7.4f} \pm {energy_std:7.1g}\) & {(np.abs(energy - exact))*100/exact:7.4f} & {gd_iterations_end} & {timing:.1f}" + r" \\")

if __name__ == "__main__":
    main()