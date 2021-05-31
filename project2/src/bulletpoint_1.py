import multiprocessing
import time
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
    proc, n_particles, n_dims = arg_list
    omega = 1
    sigma = np.sqrt(1/omega)
    
    q = ImportanceSampling(
        n_particles = n_particles,
        n_dims = n_dims,
        n_hidden = 2,
        n_mc_cycles = int(2**12),
        max_iterations = 50,
        # learning_rate = {"factor": 0.1, "init": 0.2},
        # learning_rate = 1,
        learning_rate = 0.05,
        sigma = sigma,
        interaction = False,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1]
    )
    q.initial_state(
        loc_scale_all = (0, 0.5)
    )
    q.solve(verbose=False)

    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q

def main():
    n_particles = [1, 2]
    n_dims = [1, 2, 3]
    args = []

    proc = 0
    for n_particle in n_particles:
        for n_dim in n_dims:
            args.append([proc, n_particle, n_dim])
            proc += 1

    pool = multiprocessing.Pool()
    results = pool.map(parallel, args)
    n_results = len(results)
    results_blocking = np.zeros(n_results)

    for result in range(n_results):
        data = results[result].energy_mc_iter[-1, :]
        _, results_blocking[result], _, _, _ = block(data, verbose=False)

    results_exact = ["1/2", "1", "3/2", "1", "2", "3"]

    for i in range(len(n_particles)):
        print(f"{i+1}, 1 & \({results[i*3].energies[-1]:.3f} \pm {results_blocking[i*3]:.1g}\) & {results_exact[i*3]}" + r" \\")
        print(f"{i+1}, 2 & \({results[i*3 + 1].energies[-1]:.3f} \pm {results_blocking[i*3 + 1]:.1g}\) & {results_exact[i*3 + 1]}" + r" \\")
        print(f"{i+1}, 3 & \({results[i*3 + 2].energies[-1]:.3f} \pm {results_blocking[i*3 + 2]:.1g}\) & {results_exact[i*3 + 2]}" + r" \\")

if __name__ == "__main__":
    main()