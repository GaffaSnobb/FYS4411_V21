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
    max_iterations, = arg_list
    
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
        time_step = 0.05
    )
    q.initial_state(
        loc_scale_all = (0, 2)
    )
    q.solve()

    return q

def main():
    fig, ax = plt.subplots()

    # args = [[0.06], [0.08], [0.1], [0.12], [0.14]]
    # args = [[0.11], [0.12], [0.13], [0.14]]
    args = [[20], [30], [40], [50]]
    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)
    
    for i in range(len(res)):
        print(f"{res[i].energies[-1]}", end=" ")
        ax.plot(range(res[i].max_iterations), res[i].energies, label=r"iterations: " + f"{args[i][0]}")
        ax.plot(res[i].max_iterations, res[i].energies[-1], "o")

    # ax.set_xlabel("Iterations")
    # ax.set_ylabel("Energy")
    # ax.legend()
    # plt.show()

if __name__ == "__main__":
    main()