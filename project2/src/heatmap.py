import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from boltzmann_machine import ImportanceSampling, BruteForce
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
    proc, scale, max_iteration = arg_list
    omega = 1/4
    sigma = np.sqrt(1/omega)
    
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 3,
        # n_dims = 2,
        n_hidden = 2,
        n_mc_cycles = int(2**14),
        max_iterations = max_iteration,
        learning_rate = 0.05,
        sigma = sigma,
        interaction = True,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1]
    )
    q.initial_state(
        loc_scale_all = (0, scale)
    )
    q.solve(verbose=False)

    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q

def main():
    scales = [0, 1, 2, 3, 4]
    max_iterations = [10, 20, 30, 40, 50, 60, 70]
    n_scales = len(scales)
    n_max_iterations = len(max_iterations)

    grid = np.zeros((n_scales, n_max_iterations))
    args = []
    
    for i in range(n_scales):
        for j in range(n_max_iterations):
            args.append([i*n_max_iterations + j, scales[i], max_iterations[j]])

    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)    
    res.sort(key=lambda val: val.loc_scale_all[1]*1000 + val.max_iterations)    # Trick to sort by both scale and max iterations
    
    for i in range(n_scales):
        for j in range(n_max_iterations):
            grid[i, j] = res[i*n_max_iterations + j].energies[-1]


    fig, ax = plt.subplots()
    ax = sns.heatmap(
        data = grid,
        linewidth = 0.5,
        annot = True,
        cmap = "viridis",
        ax = ax,
        yticklabels = scales,
        xticklabels = max_iterations,
        fmt = ".2f",
        annot_kws = {"size": 15}
    )
    ax.tick_params(axis='y', rotation=0)
    ax.set_xlabel(r"GD iterations")
    ax.set_ylabel(r"Initial std")
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(r"$E_L$ [a.u.]", rotation=90)
    ax.invert_yaxis()
    fig.savefig(fname="../fig/heatmap_iterations_init_scale.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()