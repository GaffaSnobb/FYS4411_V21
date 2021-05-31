import time
import multiprocessing
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
    proc, scale, learning_rate = arg_list
    omega = 1
    sigma = np.sqrt(1/omega)
    
    q = ImportanceSampling(
        n_particles = 1,
        n_dims = 1,
        n_hidden = 2,
        n_mc_cycles = int(2**12),
        max_iterations = 100,
        learning_rate = learning_rate,
        sigma = sigma,
        interaction = False,
        omega = omega,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
        rng_seed = 1337
    )
    q.initial_state(
        loc_scale_all = (0, scale)
    )
    if proc == 0:
        verbose = True
    else:
        verbose = False
    
    q.solve(verbose)

    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q

def main():
    # learning_rates = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    # learning_rates = [0.02, 0.06, 0.1, 0.14, 0.18]
    learning_rates = [0.02, 0.1, 0.18]
    n_learning_rates = len(learning_rates)
    # scales = [0.5, 1, 1.5, 2]
    scales = [0.5, 1.5]
    n_scales = len(scales)
    
    args = []
    proc = 0
    for i in range(n_scales):
        for j in range(n_learning_rates):
            args.append([proc, scales[i], learning_rates[j]])
            proc += 1
    
    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 7))
    ax = ax.ravel()
    # axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    proc = 0
    for i in range(n_scales):
        ax[i].hlines(y=0.5, xmin=0, xmax=100, label=r"Exact ($E_L = 0.5$)", linestyle="dashed", color="black")
        for j in range(n_learning_rates):
            ax[i].plot(range(res[proc].max_iterations), res[proc].energies, label=r"Learning rate: " + f"{learning_rates[j]}")
            proc += 1
        # axins.plot(range(res[i].max_iterations), res[i].energies)
        # ax.plot(res[i].max_iterations, res[i].energies[-1], "o")

    ax[0].set_xticklabels("")
    ax[0].legend(fontsize=12)
    ax[1].set_xlabel("GD iterations")
    ax[0].set_title(f"Initial STD: 0.5")
    ax[1].set_title(f"Initial STD: 1.5")
    fig.text(s=r"$E_L$ [a.u.]", x=0.02, y=0.45, rotation=90, fontsize=15)
    fig.savefig(fname="../fig/energy_vs_iterations_learning_rates.png", dpi=300)
    plt.show()

    # ax.plot([10], [10]) # Force matplotlob to label y = 10
    # ax.set_yscale("log")
    # ax.legend(fontsize=12, loc="upper left")
    # axins.set_yscale("log")
    # axins.hlines(y=0.5, xmin=0, xmax=100, label=r"Exact ($E_L = 0.5$)", linestyle="dashed", color="black")

    # axins.set_xticklabels("")
    # axins.set_yticklabels("")

    # x1, x2, y1, y2 = 80, 100, 0.48, 0.54
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # ax.indicate_inset_zoom(axins, edgecolor="black")


if __name__ == "__main__":
    main()