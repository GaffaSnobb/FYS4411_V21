import multiprocessing
import matplotlib.pyplot as plt
from boltzmann_machine import ImportanceSampling, BruteForce
import numpy as np

def parallel(arg_list: list):
    """
    This function is meant to be run in parallel with
    multiprocessing.pool.

    Parameters
    ----------
    arg_list:
        A list of arguments to pass to the class constructor.
    """

    if arg_list[0] == "importance":

        n_particles, n_dims, n_hidden, n_mc_cycles, max_iterations, learning_rate, sigma, interaction, diffusion_coeff, time_step, = arg_list[1:]

        q = ImportanceSampling(
            n_particles     = n_particles,
            n_dims          = n_dims,
            n_hidden        = n_hidden,
            n_mc_cycles     = n_mc_cycles,
            max_iterations  = max_iterations,
            learning_rate   = learning_rate,
            sigma           = sigma,
            interaction     = interaction,
            diffusion_coeff = diffusion_coeff,
            time_step       = time_step
        )

    elif arg_list[0] == "brute":

        n_particles, n_dims, n_hidden, n_mc_cycles, max_iterations, learning_rate, sigma, interaction, brute_force_step_size, = arg_list[1:]

        q = BruteForce(
            n_particles           = n_particles,
            n_dims                = n_dims,
            n_hidden              = n_hidden,
            n_mc_cycles           = n_mc_cycles,
            max_iterations        = max_iterations,
            learning_rate         = learning_rate,
            sigma                 = sigma,
            interaction           = interaction,
            brute_force_step_size = brute_force_step_size
        )


    q.initial_state(loc_scale_all = (0, 1))

    q.solve(verbose=False)

    return q


def main(type = "importance", PATH=""):

    # Common parameters for both sampling methods
    n_particles     = 2
    n_dims          = 2
    n_hidden        = 2
    n_mc_cycles     = int(2**12)
    max_iterations  = 50
    sigma           = 1
    interaction     = True
    learning_rates  = [0.02, 0.04, 0.06, 0.08, 0.1]

    # Only for importance sampling
    diffusion_coeff = 0.5
    time_step       = 0.05

    # Only for brute force sampling
    brute_force_step_size = 0.05

    fnames = []
    args = []

    # Set up the arguments to be passed to paralell()
    for lr in learning_rates:

        if type == "importance":
            list = [type, n_particles, n_dims, n_hidden, n_mc_cycles, max_iterations, lr, sigma, interaction, diffusion_coeff, time_step]

        elif type == "brute":
            list = [type, n_particles, n_dims, n_hidden, n_mc_cycles, max_iterations, lr, sigma, interaction, brute_force_step_size]

        args.append(list)
        fnames.append(f"{PATH}/{type}_{interaction}_{lr}")


    pool = multiprocessing.Pool()
    res = pool.map(parallel, args)

    # Save energy_mc_iter to file
    print()
    for i, name in enumerate(fnames):
        np.save(f"{name}", res[i].energy_mc_iter)
        print(f"saved as: {name}.npz")


if __name__ == "__main__":
    main(type="importance", PATH="../out")
    main(type="brute", PATH="../out")
