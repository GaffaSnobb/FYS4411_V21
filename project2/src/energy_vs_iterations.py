import time
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from boltzmann_machine import BruteForce
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
    
    q = BruteForce(
        n_particles = 1,
        n_dims = 1,
        n_hidden = 2,
        n_mc_cycles = int(2**18),
        max_iterations = 12,
        learning_rate = learning_rate,
        sigma = sigma,
        interaction = False,
        omega = omega,
        brute_force_step_size = 1,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1],
        rng_seed = 1337+4
    )
    q.initial_state(
        loc_scale_all = (0, scale)
    )    
    
    q.solve(
        verbose = True if proc == 0 else False,
        save_state = True,
        load_state = True
    )

    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q

def main():
    """
    Plot local energy as a function of gradient descent iterations for
    different learning rates. Include variable learning rate.
    """
    learning_rates = [0.5, {"factor": 0.05, "init": 0.5}, 0.6, {"factor": 0.05, "init": 0.6}]
    scales = [0.5, 4]
    n_learning_rates = len(learning_rates)
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
        ax[i].hlines(
            y = 0.5,
            xmin = 0,
            xmax = 12,
            label = r"Exact ($E_L = 0.5$ a.u.)",
            linestyle = "dashed",
            color = "black"
        )
        
        for j in range(n_learning_rates):
            label = r"$\eta$ = " + f"{learning_rates[j]}"
            linestyle = "solid"
            
            if isinstance(learning_rates[j], dict):
                label = r"$\eta_{init} = $" + f"{learning_rates[j]['init']}"
                label += r" $f_{\eta} = $" + f"{learning_rates[j]['factor']}"
                linestyle = "dashed"
            
            ax[i].plot(
                range(res[proc].max_iterations),
                res[proc].energies,
                label = label,
                linestyle = linestyle
            )
            proc += 1

    ax[0].set_xticklabels("")
    ax[0].legend(fontsize=12)
    ax[1].set_xlabel("GD iterations")
    ax[0].set_title(r"$\sigma_{init} = $" + f"{res[0].loc_scale_all[1]}")
    ax[1].set_title(r"$\sigma_{init} = $" + f"{res[-1].loc_scale_all[1]}")
    fig.text(s=r"$E_L$ [a.u.]", x=0.02, y=0.45, rotation=90, fontsize=15)
    fig.savefig(fname="../fig/energy_vs_iterations_learning_rates.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()