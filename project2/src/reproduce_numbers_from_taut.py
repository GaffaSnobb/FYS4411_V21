import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
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
    proc, omega, sigma = arg_list
    
    q = BruteForce(
        n_particles = 2,
        n_dims = 3,
        n_hidden = 4,
        n_mc_cycles = int(2**12),
        max_iterations = 50,
        learning_rate = {"init": 0.6, "factor": 0.05},
        sigma = sigma,
        interaction = True,
        brute_force_step_size = 1,
        omega = omega,
        parent_data_directory = (__file__.split(".")[0]).split("/")[-1]
    )
    q.initial_state(
        loc_scale_all = (0, 1)
    )
    q.solve(
        verbose = True if proc == 0 else False,
        save_state = True,
        load_state = True,
        calculate_blocking_all = False
    )
    print(f"Process {proc} finished in {time.time() - timing:.3f}s with parameters {arg_list[1:]}")

    return q

def main():
    """
    Reproduce numbers from figure 1 and table 1 in
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.48.3561.

    3D system with 2 particles for different values of the potential
    oscillator frequency, omega. The relation sigma = sqrt(1/omega) is
    used.
    """
    fig, ax = plt.subplots()

    energies_taut = np.array([0.625, 0.175, 0.0822, 0.0477, 0.0311, 0.0219, 0.0162, 0.0125, 0.01, 0.0081])*2
    omegas_taut_inverse = np.array([4, 20, 54.7386, 115.299, 208.803, 342.366, 523.102, 758.124, 1054.54, 1419.47])
    omegas_taut = 1/omegas_taut_inverse
    
    args = [[proc, omega, np.sqrt(1/omega)] for proc, omega in enumerate(omegas_taut)]
    pool = multiprocessing.Pool()
    results = pool.map(parallel, args)
    n_results = len(results)
    energies = np.zeros(n_results)
    
    print()
    # for omega, our, taut, diff in zip(omegas_taut_inverse, energies, energies_taut, np.abs(energies - energies_taut)):
    for i in range(n_results):
        """
        Print the values in LaTeX table format.
        """
        omega = omegas_taut_inverse[i]
        our = results[i].energies[-1]
        energies[i] = results[i].energies[-1]   # For the plot
        our_std = results[i].blocking_final[0]
        taut = energies_taut[i]
        # diff = abs(our - taut)*100/taut
        diff = abs(our - taut)

        print(f"{omega} & \({our:.5f} \pm {our_std:.5f} \) & {taut} & {diff:.5f}" + r" \\")

    ax.plot(1/omegas_taut, energies_taut, "--.", label="M. Taut")
    ax.plot(1/omegas_taut, energies, "--.", label="This work")
    ax.set_xlabel(r"1/$\omega$")
    ax.set_ylabel(r"$E_L$ [a.u.]")
    ax.legend()
    ax.tick_params()
    fig.savefig(fname="../fig/tune_omega_taut_vs_this_work.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()
    pass