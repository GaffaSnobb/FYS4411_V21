import multiprocessing
from boltzmann_machine import BruteForce, ImportanceSampling

def rng_seed_parallel(rng_seed):
    q = ImportanceSampling(
        n_particles = 1,
        n_dims = 1,
        n_hidden = 1,
        n_mc_cycles = 1,
        max_iterations = 1,
        learning_rate = 1,
        sigma = 1,
        interaction = False,
        omega = 1,
        diffusion_coeff = 1,
        time_step = 1,
        parent_data_directory = None,
        rng_seed = rng_seed
    )
    return q

def test_rng_seed_different():
    """
    Test that multiprocessing threads draw different random numbers.
    """
    pool = multiprocessing.Pool()
    args = [None, None]
    results = pool.map(rng_seed_parallel, args)

    success = results[0].visible_biases != results[1].visible_biases
    msg = "Warning! Same RNG seed for two individual runs!"
    assert success, msg

def test_rng_seed_equal():
    """
    Test that multiprocessing threads draw equal random numbers when
    given the same rng seed.
    """
    pool = multiprocessing.Pool()
    args = [1337, 1337]
    results = pool.map(rng_seed_parallel, args)

    success = results[0].visible_biases == results[1].visible_biases
    msg = "Warning! Same RNG seed does not produce the same random numbers!"
    assert success, msg

def known_values_parallel(arg_list: list):
    proc, n_particles, n_dims, method = arg_list
    
    if method == "importance":
        q = ImportanceSampling(
            n_particles = n_particles,
            n_dims = n_dims,
            n_hidden = 2,
            n_mc_cycles = int(2**11),
            max_iterations = 50,
            learning_rate = 0.05,
            sigma = 1,
            interaction = False,
            omega = 1,
            diffusion_coeff = 0.5,
            time_step = 0.05,
            parent_data_directory = None
        )
    elif method == "brute":
        q = BruteForce(
            n_particles = n_particles,
            n_dims = n_dims,
            n_hidden = 2,
            n_mc_cycles = int(2**11),
            max_iterations = 50,
            learning_rate = 0.05,
            sigma = 1,
            interaction = False,
            omega = 1,
            brute_force_step_size = 1,
            parent_data_directory = None
        )
    q.initial_state(
        loc_scale_all = (0, 0.5)
    )
    q.solve(
        verbose = False,
        save_state = False,
        calculate_blocking_all = False
    )

    return q

def test_known_values():
    """
    Compare calculated values to known test values for 1, 2, 3 dims
    with 1 and 2 particles. No interaction.
    """
    methods = ["brute", "importance"]
    n_particles = [1, 2]
    n_dims = [1, 2, 3]
    args = []

    proc = 0
    for method in methods:
        for n_particle in n_particles:
            for n_dim in n_dims:
                args.append([proc, n_particle, n_dim, method])
                proc += 1

    pool = multiprocessing.Pool()
    results = pool.map(known_values_parallel, args)

    results_exact = [0.5, 1, 1.5, 1, 2, 3]*2
    tol = 0.1

    fail_counter = 0
    i = 0
    while i < len(results):
        success = abs(results[i].energies[-1] - results_exact[i]) < tol
        msg = "Warning! Deviation from known analytical result"
        msg += f" for method {results[i].__str__()}!"
        msg += f" Expected {results_exact[i]} got {results[i].energies[-1]}"
        msg += f" (proc {args[i][0]})."
        
        if (not success) and (fail_counter < 2):
            """
            Allow two fails, due to the stochastic nature of VMC.
            """
            results[i] = known_values_parallel(args[i])
            fail_counter += 1
            continue

        assert success, msg
        i += 1

if __name__ == "__main__":
    print("Please use pytest to run these tests!")