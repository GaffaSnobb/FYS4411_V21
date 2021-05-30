import time
import multiprocessing
import matplotlib.pyplot as plt
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
    proc, max_iterations = arg_list
    omega = 1/4
    sigma = np.sqrt(1/omega)
    
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 2,
        n_hidden = 2,
        n_mc_cycles = int(2**13),
        max_iterations = max_iterations,
        learning_rate = 0.05,
        sigma = 1,              # Std. of the normal distribution the visible nodes.
        interaction = True,
        omega = 1,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
    )
    q.initial_state(
        loc_scale_all = (0, 2)
    )
    q.solve()

    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q

def main():
    
    max_iterations = [10, 20, 30, 40, 50]
    
    args = []
    for i in range(len(max_iterations)):
        args.append([i, max_iterations[i]])
    
    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)

    fig, ax = plt.subplots()
    for i in range(len(res)):
        ax.plot(range(res[i].max_iterations), res[i].energies, label=r"iterations: " + f"{args[i][0]}")
        ax.plot(res[i].max_iterations, res[i].energies[-1], "o")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energy")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()