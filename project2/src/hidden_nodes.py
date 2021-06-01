import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from boltzmann_machine import ImportanceSampling

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
        max_iterations = 50,
        learning_rate = 0.05,
        sigma = sigma,
        interaction = False,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
        rng_seed = 1337 
    )
    q.initial_state(
        loc_scale_all = (0, 1)
    )
    q.solve(verbose=False, save_state=True)

    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q

def main():
    fig, ax = plt.subplots()

    n_hidden_nodes = [2, 3, 4, 5]
    args = []

    for i in range(len(n_hidden_nodes)):
        args.append([i, n_hidden_nodes[i]])

    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)
    
    for i in range(len(res)):
        ax.plot(range(res[i].max_iterations), res[i].energies, label=r"n_hidden: " + f"{args[i][1]}")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energy")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()