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
    proc, loc_scale_all = arg_list
    
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 2,
        n_hidden = 2,
        n_mc_cycles = int(2**14),
        max_iterations = 50,
        learning_rate = 0.05,
        sigma = 1,
        interaction = True,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        omega = 1
    )
    q.initial_state(
        loc_scale_all = loc_scale_all
    )
    if proc == 0:
        q.solve(verbose=True)
    else:
        q.solve(verbose=False)

    return q

def main():
    fig, ax = plt.subplots()

    args = [[0, (0, 0)], [1, (0, 0.1)], [2, (0, 1)]]
    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)

    for i in range(len(res)):
        ax.plot(range(res[i].max_iterations), res[i].energies, label="(loc, scale): " + f"{args[i][1]}")

    ax.set_xlabel("GD iterations")
    ax.set_ylabel("Energy [a.u.]")
    ax.legend()
    ax.tick_params()
    plt.show()

if __name__ == "__main__":
    main()