import multiprocessing
import numpy as np
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
    proc, omega, sigma = arg_list
    
    q = ImportanceSampling(
        n_particles = 2,
        n_dims = 3,
        n_hidden = 2,
        n_mc_cycles = int(2**14),
        max_iterations = 50,
        learning_rate = 0.05,
        sigma = sigma,
        interaction = True,
        diffusion_coeff = 0.5,
        time_step = 0.05,
        omega = omega
    )
    q.initial_state(
        loc_scale_all = (0, 0.1)
    )
    if proc == 0:
        q.solve(verbose=True)
    else:
        q.solve(verbose=False)

    return q

def main():
    fig, ax = plt.subplots()

    energies_taut = np.array([0.625, 0.175, 0.0822, 0.0477, 0.0311, 0.0219, 0.0162, 0.0125, 0.01, 0.0081])*2
    omegas_taut_inverse = np.array([4, 20, 54.7386, 115.299, 208.803, 342.366, 523.102, 758.124, 1054.54, 1419.47])
    omegas_taut = 1/omegas_taut_inverse
    # omegas = np.arange(1, 6+1, 1)
    
    args = [[proc, omega, np.sqrt(1/omega)] for proc, omega in enumerate(omegas_taut)]
    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)

    energies = np.zeros(len(res))

    for i in range(len(res)):
        energies[i] = res[i].energies[-1]
    
    for omega, our, taut, diff in zip(omegas_taut_inverse, energies, energies_taut, np.abs(energies - energies_taut)):
        """
        Print the values in LaTeX table format.
        """
        print(f"{omega} & {our:.4f} & {taut:.4f} & {diff:.4f}" + r" \\")

    ax.plot(1/omegas_taut, energies_taut, "--.", label="M. Taut")
    ax.plot(1/omegas_taut, energies, "--.", label="This work")
    # ax.set_title("3D")
    ax.set_xlabel(r"1/$\omega$")
    ax.set_ylabel("Energy [a.u.]")
    ax.legend()
    ax.tick_params()
    fig.savefig(fname="../fig/tune_omega_taut_vs_this_work.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    pass