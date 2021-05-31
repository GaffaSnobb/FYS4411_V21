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
    """
    Plot local energy as a function of gradient descent iterations for
    different learning rates. Include variable learning rate.
    """
    # learning_rates = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    # learning_rates = [0.02, 0.06, 0.1, 0.14, 0.18]
    learning_rates = [0.02, 0.1, 0.18, {"factor": 0.05, "init": 0.18}]
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
    proc = 0
    for i in range(n_scales):
        ax[i].hlines(y=0.5, xmin=0, xmax=100, label=r"Exact ($E_L = 0.5$ a.u.)", linestyle="dashed", color="black")
        for j in range(n_learning_rates):
            label = r"$\eta$ = " + f"{learning_rates[j]}"
            if isinstance(learning_rates[j], dict):
                label = r"$\eta_{init} = $" + f"{learning_rates[j]['init']}"
                label += r" $f_{\eta} = $" + f"{learning_rates[j]['factor']}"
            ax[i].plot(range(res[proc].max_iterations), res[proc].energies, label=label)
            proc += 1

    ax[0].set_xticklabels("")
    ax[0].legend(fontsize=12)
    ax[1].set_xlabel("GD iterations")
    ax[0].set_title(r"$\sigma_{init} = 0.5$")
    ax[1].set_title(r"$\sigma_{init} = 1.5$")
    fig.text(s=r"$E_L$ [a.u.]", x=0.02, y=0.45, rotation=90, fontsize=15)
    fig.savefig(fname="../fig/energy_vs_iterations_learning_rates.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()