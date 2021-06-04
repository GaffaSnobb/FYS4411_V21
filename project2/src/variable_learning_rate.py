import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from boltzmann_machine import ImportanceSampling
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
    proc, learning_rate = arg_list
    omega = 1
    sigma = np.sqrt(1/omega)
    
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 3,
        n_hidden = 2,
        n_mc_cycles = int(2**14),
        max_iterations = 50,
        learning_rate = learning_rate,
        sigma = sigma,
        interaction = True,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
        rng_seed = 1337
    )
    q.initial_state(
        loc_scale_all = (0, 2)
    )

    q.solve(
        verbose = True if proc == 0 else False,
        save_state = True,
        load_state = True
    )

    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q

def main():
    fig, ax = plt.subplots()

    # learning_rates = [0.05, {"factor": 1, "t0": 2.5, "t1": 50}, {"factor": 2, "t0": 2.5, "t1": 50}, {"factor": 3, "t0": 2.5, "t1": 50}, {"factor": 4, "t0": 2.5, "t1": 50}]
    learning_rate_init = 0.5

    # learning_rates = [
    #     learning_rate_init,
    #     {"factor": 0.2, "init": learning_rate_init},
    #     {"factor": 0.4, "init": learning_rate_init},
    #     {"factor": 0.6, "init": learning_rate_init},
    #     {"factor": 0.8, "init": learning_rate_init},
    #     {"factor": 1.0, "init": learning_rate_init},
    # ]
    learning_rates = [
        learning_rate_init,
        {"factor": 0.005, "init": learning_rate_init},
        {"factor": 0.010, "init": learning_rate_init},
        {"factor": 0.015, "init": learning_rate_init},
        {"factor": 0.020, "init": learning_rate_init},
        {"factor": 0.025, "init": learning_rate_init},
    ]
    args = []

    for i in range(len(learning_rates)):
        args.append([i, learning_rates[i]])

    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)
    
    for i in range(len(res)):
        ax.plot(range(res[i].max_iterations), res[i].energies, label=r"$\eta$: " + f"{args[i][1]}")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energy")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()