import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    scale, max_iteration = arg_list
    
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 2,
        n_hidden = 2,
        n_mc_cycles = int(2**14),
        max_iterations = max_iteration,
        learning_rate = 0.05,
        sigma = 1,              # Std. of the normal distribution the visible nodes.
        interaction = True,
        omega = 1,
        diffusion_coeff = 0.5,
        time_step = 0.05
    )
    q.initial_state(
        loc_scale_all = (0, scale)
    )
    q.solve()

    return q

def main():

    scales = [0.5, 1, 1.5, 2]
    max_iterations = [20, 30, 40, 50]
    max_iterations_reversed = list(reversed(max_iterations))
    n_scales = len(scales)
    n_max_iterations = len(max_iterations)

    grid = np.zeros((n_scales, n_max_iterations))
    args = []
    
    for i in range(n_scales):
        for j in range(n_max_iterations):
            args.append([scales[i], max_iterations[j]])

    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)
    # np.random.shuffle(res)
    
    res.sort(key=lambda val: val.loc_scale_all[1]*1000 + val.max_iterations)

    # for val in res:
    #     print(val.max_iterations, val.loc_scale_all[1])
    
    for i in range(n_scales):
        for j in range(n_max_iterations):
            grid[i, j] = res[i*n_scales + j].energies[-1]

    print(grid[-1, :])

    # fig, ax = plt.subplots()
    # ax = sns.heatmap(
    #     data = grid,
    #     linewidth = 0.5,
    #     annot = True,
    #     cmap = "viridis",
    #     ax = ax,
    #     xticklabels = scales,
    #     yticklabels = max_iterations
    # )
    # # ax.invert_yaxis()
    
    # plt.show()

if __name__ == "__main__":
    main()