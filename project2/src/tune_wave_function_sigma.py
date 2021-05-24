import multiprocessing
import matplotlib.pyplot as plt
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
    sigma, = arg_list
    
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 2,
        n_hidden = 2,
        n_mc_cycles = int(2**12),
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
    q.solve()

    return q

def main():
    fig, ax = plt.subplots()

    args = [[0.8], [0.9], [1.] , [1.1], [1.2]]
    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)

    for i in range(len(res)):
        ax.plot(range(res[i].max_iterations), res[i].energies, label=r"$\sigma: $" + f"{args[i][0]}")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Energy")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()