import multiprocessing
import matplotlib.pyplot as plt
from boltzmann_machine import ImportanceSampling, BruteForce
import mplrc_params

def parallel(arg_list: list):
    """
    This function is meant to be run in parallel with
    multiprocessing.pool.

    Parameters
    ----------
    arg_list:
        A list of arguments to pass to the class constructor.
    """
    proc, sigma = arg_list
    
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 2,
        n_hidden = 2,
        n_mc_cycles = int(2**14),
        max_iterations = 50,
        learning_rate = 0.05,
        sigma = sigma,
        interaction = True,
        diffusion_coeff = 0.5,
        time_step = 0.05
    )
    q.initial_state(
        loc_scale_all = (0, 1)
    )
    if proc == 0:
        q.solve(verbose=True)
    else:
        q.solve(verbose=False)

    return q

def main():
    fig, ax = plt.subplots()

    # args = [[0, 0.6], [1, 0.8], [2, 0.9], [3, 1.] , [4, 1.1], [5, 1.2], [6, 1.4], [7, 1.6]]
    args = [[0, 0.6], [2, 0.9], [4, 1.1], [6, 1.4]]
    # args = [[0, 0.5] , [1, 0.75], [2, 1.], [3, 1.25], [4, 1.5]]
    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)

    for i in range(len(res)):
        ax.plot(range(res[i].max_iterations), res[i].energies, label=r"$\sigma: $" + f"{args[i][1]}")

    ax.set_xlabel("GD iterations")
    ax.set_ylabel("Energy [a.u.]")
    ax.legend()
    ax.tick_params()
    plt.show()

if __name__ == "__main__":
    main()