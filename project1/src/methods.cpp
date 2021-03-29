#include "methods.h"

BruteForce::BruteForce(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input,
    arma::Col<double> alphas_input,
    const double beta_input,
    const double brute_force_step_size_input,
    const bool numerical_differentiation_input,
    bool debug_input
) : VMC(
        n_dims_input,
        n_variations_input,
        n_mc_cycles_input,
        n_particles_input,
        alphas_input,
        beta_input,
        numerical_differentiation_input,
        debug_input
    ),
    step_size(brute_force_step_size_input)
{   /*
    Class constructor.

    Parameters
    ----------
    n_dims_input : constant integer
        The number of spatial dimensions.

    n_variations_input : constant integer
        The number of variational parameters.

    n_mc_cycles_input : constant integer
        The number of Monte Carlo cycles per variational parameter.

    n_particles_input : constant integer
        The number of particles.

    alphas_input : armadillo column vector
        A linspace of the variational parameters.

    brute_force_step_size_input : constant double
        The step size for when new random positions are drawn for the
        brute force approach.
    */
}

void BruteForce::one_variation(int variation)
{   /*
    Perform calculations for a single variational parameter.

    Parameters
    ----------
    variation : int
        Which iteration of variational parameter alpha.
    */

    double alpha = alphas(variation);
    int acceptance = 0;  // Debug.

    energy_expectation = 0; // Reset for each variation.
    energy_variance = 0;    // Reset for each variation. NB: Variable not inside parallel region.
    energy_expectation_squared = 0;

    // One-body density.
    double particle_distance;
    particle_per_bin_count_thread.zeros();
    // One-body density end.

    for (particle = 0; particle < n_particles; particle++)
    {   /*
        Iterate over all particles.  In this loop, all current
        positions are calulated along with the current wave
        functions.
        */
        for (dim = 0; dim < n_dims; dim++)
        {   /*
            Set initial values.
            */
            pos_current(dim, particle) = step_size*(uniform(engine) - 0.5);
        }
    }
    wave_current = wave_function_ptr(
        pos_current,  // Position of one particle.
        alpha,
        beta,
        n_particles
    );

    #pragma omp parallel \
        private(mc, particle, dim, particle_inner) \
        private(wave_new) \
        firstprivate(wave_current, local_energy) \
        firstprivate(pos_new, pos_current, particle_per_bin_count_thread) \
        reduction(+:acceptance, energy_expectation, energy_expectation_squared) \
        private(engine)
    {
        #ifdef _OPENMP
            engine.seed(seed + omp_get_thread_num());
        #endif

        #pragma omp for
        for (mc = 0; mc < n_mc_cycles; mc++)
        {   /*
            Run over all Monte Carlo cycles.
            */
            for (particle = 0; particle < n_particles; particle++)
            {   /*
                Iterate over all particles.  In this loop, new
                proposed positions and wave functions are
                calculated.
                */
                for (dim = 0; dim < n_dims; dim++)
                {   /*
                    Set new values.
                    */
                    pos_new(dim, particle) = pos_current(dim, particle) + step_size*(uniform(engine) - 0.5);
                }

                wave_new = wave_function_ptr(
                        pos_new,  // Particle positions.
                        alpha,
                        beta,
                        n_particles
                    );

                double wave_ratio = wave_new/wave_current;
                wave_ratio *= wave_ratio;

                if (uniform(engine) < wave_ratio)
                {   /*
                    Perform the Metropolis algorithm.
                    */

                    acceptance++;    // Debug.
                    pos_current.col(particle) = pos_new.col(particle);
                    wave_current = wave_new;

                    local_energy = 0;   // Overwrite local energy from previous particle step.
                    for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                    {   /*
                        After moving one particle, the local energy is
                        calculated based on all particle positions.
                        */
                        local_energy += local_energy_ptr(
                            pos_current,
                            alpha,
                            beta,
                            particle_inner,
                            n_particles
                        );
                    }
                    // One-body density.
                    particle_distance = arma::norm(pos_current.col(particle), 2);
                    for (bin = 0; bin < n_bins - 1; bin++)
                    {
                        if (
                            (particle_distance >= bin_locations(bin)) and
                            (particle_distance <  bin_locations(bin + 1))
                        )
                        {
                            particle_per_bin_count_thread(bin) += 1;
                            break;  // No need to continue checking for this particle!
                        }
                    }
                    // One-body density end.
                }
                energy_expectation += local_energy;
                energy_expectation_squared += local_energy*local_energy;
            }
            energies(mc, variation) = local_energy;
        }
        #pragma omp critical
        {
            particle_per_bin_count.col(variation) += particle_per_bin_count_thread;
        }
    }   // Parallel end.

    energy_expectation /= n_mc_cycles;
    energy_expectation_squared /= n_mc_cycles;
    energy_expectation /= n_particles;
    energy_expectation_squared /= n_particles;
    energy_variance = energy_expectation_squared
        - energy_expectation*energy_expectation;

    acceptances(variation) = acceptance;    // Debug.
}

ImportanceSampling::ImportanceSampling(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input,
    arma::Col<double> alphas_input,
    const double beta_input,
    const double importance_time_step_input,
    const bool numerical_differentiation_input,
    bool debug_input
) : VMC(
        n_dims_input,
        n_variations_input,
        n_mc_cycles_input,
        n_particles_input,
        alphas_input,
        beta_input,
        numerical_differentiation_input,
        debug_input
    ),
    time_step(importance_time_step_input)
{   /*
    Class constructor.

    Parameters
    ----------
    n_dims_input : constant integer
        The number of spatial dimensions.

    n_variations_input : constant integer
        The number of variational parameters.

    n_mc_cycles_input : constant integer
        The number of Monte Carlo cycles per variational parameter.

    n_particles_input : constant integer
        The number of particles.

    alphas_input : armadillo column vector
        A linspace of the variational parameters.

    importance_time_step_input : constant double

    */
}

void ImportanceSampling::one_variation(int variation)
{   /*
    Perform calculations for a single variational parameter.

    Parameters
    ----------
    variation : int
        Which iteration of variational parameter alpha.
    */

    double alpha = alphas(variation);
    int acceptance = 0;  // Debug. Count the number of accepted steps.

    // Reset values for each variation.
    energy_expectation = 0;
    energy_variance = 0;
    energy_expectation_squared = 0;

    // GD specifics.
    wave_derivative = 0;
    wave_derivative_expectation = 0;
    wave_times_energy_expectation = 0;
    // GD specifics end.

    // One-body density.
    double particle_distance;
    particle_per_bin_count_thread.zeros();
    // One-body density end.
    for (particle = 0; particle < n_particles; particle++)
    {   /*
        Iterate over all particles.  In this loop, all current
        positions are calulated along with the current wave
        functions.
        */

        for (dim = 0; dim < n_dims; dim++)
        {   /*
            Set initial values.
            */
            pos_current(dim, particle) = normal(engine)*sqrt(time_step);
        }

        qforce_current.col(particle) = quantum_force_ptr(
            pos_current,
            alpha,
            beta,
            particle,
            n_particles
        );
    }

    wave_current = wave_function_ptr(
        pos_current,  // Position of all particles.
        alpha,
        beta,
        n_particles
    );

    #pragma omp parallel\
        private(mc, particle, dim, particle_inner, bin) \
        private(wave_new) \
        firstprivate(wave_current, local_energy) \
        firstprivate(pos_new, qforce_new, pos_current, qforce_current) \
        reduction(+:acceptance, energy_expectation, energy_expectation_squared) \
        reduction(+:wave_times_energy_expectation, wave_derivative_expectation) \
        firstprivate(wave_derivative, particle_per_bin_count_thread) \
        private(engine)
    {
        #ifdef _OPENMP
            engine.seed(seed + omp_get_thread_num());
        #endif

        #pragma omp for
        for (mc = 0; mc < n_mc_cycles; mc++)
        {   /*
            Run over all Monte Carlo cycles.
            */
            for (particle = 0; particle < n_particles; particle++)
            {   /*
                Iterate over all particles.  In this loop, new
                proposed positions are calculated.
                */
                for (dim = 0; dim < n_dims; dim++)
                {   /*
                    Set new positions.
                    */
                    pos_new(dim, particle) = pos_current(dim, particle) +
                        diffusion_coeff*qforce_current(dim, particle)*time_step +
                        normal(engine)*sqrt(time_step);
                }

                qforce_new.col(particle) = quantum_force_ptr(
                    pos_new,
                    alpha,
                    beta,
                    particle,
                    n_particles
                );

                wave_new = wave_function_ptr(
                        pos_new,  // Particle positions.
                        alpha,
                        beta,
                        n_particles
                    );

                double greens_ratio = 0;
                for (dim = 0; dim < n_dims; dim++)
                {   /*
                    Calculate greens ratio for the acceptance criterion.
                    */
                    greens_ratio +=
                        0.5*(qforce_current(dim, particle) + qforce_new(dim, particle))
                        *(0.5*diffusion_coeff*time_step*
                        (qforce_current(dim, particle) - qforce_new(dim, particle))
                        - pos_new(dim, particle) + pos_current(dim, particle));
                }

                greens_ratio = exp(greens_ratio);

                double wave_ratio = wave_new/wave_current;
                wave_ratio *= wave_ratio;   // TODO: Find out why we need wave_ratio**2.

                if (uniform(engine) < greens_ratio*wave_ratio)
                {   /*
                    Metropolis check.
                    */
                    acceptance++;    // Debug.
                    pos_current.col(particle) = pos_new.col(particle);
                    qforce_current.col(particle) = qforce_new.col(particle);
                    wave_current = wave_new;

                    local_energy = 0;   // Overwrite local energy from previous particle step.
                    for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                    {   /*
                        After moving one particle, the local energy is
                        calculated based on all particle positions.
                        */
                        local_energy += local_energy_ptr(
                            pos_current,
                            alpha,
                            beta,
                            particle_inner,
                            n_particles
                        );
                    }
                    wave_derivative = 0;
                    for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                    {   /*
                        Calculations needed for gradient descent.
                        */
                        wave_derivative += wave_function_diff_wrt_alpha_ptr(
                            pos_current.col(particle_inner),
                            alpha,
                            beta
                        );
                    }
                }

                // GD specifics.
                wave_derivative_expectation += wave_derivative;
                wave_times_energy_expectation += wave_derivative*local_energy;
                // GD specifics end.

                // One-body density.
                particle_distance = arma::norm(pos_current.col(particle), 2);
                for (bin = 0; bin < n_bins - 1; bin++)
                {
                    if (
                        (particle_distance >= bin_locations(bin)) and
                        (particle_distance <  bin_locations(bin + 1))
                    )
                    {
                        particle_per_bin_count_thread(bin) += 1;
                        break;  // No need to continue checking for this particle!
                    }
                }
                // One-body density end.
                energy_expectation += local_energy;
                energy_expectation_squared += local_energy*local_energy;
            }
            energies(mc, variation) = local_energy;
        }
        #pragma omp critical
        {
            particle_per_bin_count.col(variation) += particle_per_bin_count_thread;
        }
    }   // Parallel end.

    acceptances(variation) = acceptance;    // Debug.
    energy_expectation /= n_mc_cycles;
    energy_expectation /= n_particles;
    energy_expectation_squared /= n_particles;
    energy_expectation_squared /= n_mc_cycles;
    energy_variance = energy_expectation_squared
        - energy_expectation*energy_expectation;

    // GD specifics.
    wave_times_energy_expectation /= n_mc_cycles;
    wave_derivative_expectation /= n_mc_cycles;
    // GD specifics end.
}

GradientDescent::GradientDescent(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input,
    const double importance_time_step_input,
    const double learning_rate_input,
    const double initial_alpha_input,
    const double beta_input,
    const bool numerical_differentiation_input,
    bool debug_input
) : ImportanceSampling(
        n_dims_input,
        n_variations_input,
        n_mc_cycles_input,
        n_particles_input,
        arma::linspace(0, 0, n_variations_input),   // Dummy input. Not in use.
        beta_input,
        importance_time_step_input,
        numerical_differentiation_input,
        debug_input
    ),
    learning_rate(learning_rate_input),
    initial_alpha(initial_alpha_input)
{   /*
    Class constructor.

    Parameters
    ----------
    n_dims_input : constant integer
        The number of spatial dimensions.

    n_variations_input : constant integer
        The number of variational parameters.

    n_mc_cycles_input : constant integer
        The number of Monte Carlo cycles per variational parameter.

    n_particles_input : constant integer
        The number of particles.

    alphas_input : armadillo column vector
        A linspace of the variational parameters.

    importance_time_step_input : constant double

    */
}

void GradientDescent::solve(const double tol)
{   /*
    Iterate over variational parameters.  Use gradient descent to
    efficiently calculate alphas.

    Parameters
    ----------
    tol : constant double
        Tolerance for the GD cutoff.
    */
    if (!call_set_local_energy)
    {
        std::cout << "Local energy is not set! Exiting..." << std::endl;
        exit(0);
    }

    if (!call_set_wave_function)
    {
        std::cout << "Wave function is not set! Exiting..." << std::endl;
        exit(0);
    }

    if (!call_set_quantum_force)
    {
        std::cout << "Quantum force is not set! Exiting..." << std::endl;
        exit(0);
    }
    double energy_derivative = 0;
    alphas(0) = initial_alpha;
    double comp_time;

    #ifdef _OPENMP
        double t1;
        double t2;
    #else
        std::chrono::steady_clock::time_point t1;
        std::chrono::steady_clock::time_point t2;
        std::chrono::duration<double> comp_time_chrono;
    #endif

    for (int variation = 0; variation < n_variations - 1; variation++)
    {
        #ifdef _OPENMP
            t1 = omp_get_wtime();
        #else
            t1 = std::chrono::steady_clock::now();
        #endif

        one_variation(variation);
        e_expectations(variation) = energy_expectation;
        e_variances(variation) = energy_variance;

        energy_derivative = 2*(wave_times_energy_expectation -
            wave_derivative_expectation*energy_expectation);

        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time_chrono = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            comp_time = comp_time_chrono.count();
        #endif

        alphas(variation + 1) = alphas(variation) - learning_rate*energy_derivative;

        std::cout << "variation : " << std::setw(3) <<  variation;
        std::cout << ", alpha: " << std::setw(10) << alphas(variation);
        std::cout << ", energy: " << std::setw(10) << energy_expectation;
        std::cout << ", variance: " << std::setw(10) << energy_variance;
        std::cout << ", acceptance: " << std::setw(7) << acceptances(variation)/(n_mc_cycles*n_particles);
        std::cout << ",  time : " << comp_time << "s" << std::endl;
        timing(variation) = comp_time;

        if (debug)
        {
            std::cout << "energy_expectation: " << energy_expectation << std::endl;
            std::cout << "wave_derivative_expectation: " << wave_derivative_expectation << std::endl;
            std::cout << "wave_derivative_expectation*energy_expectation/n_particles: " << wave_derivative_expectation*energy_expectation/n_particles << std::endl;
            std::cout << "wave_times_energy_expectation: " << wave_times_energy_expectation << std::endl;
            std::cout << "energy_derivative: " << energy_derivative << std::endl;
            std::cout << "\n";
        }
        if (variation >= 4)
        {   /*
            Run at least a few variations before breaking.
            */
            if (std::abs(alphas(variation + 1) - alphas(variation - 4)) < tol)
            {
                n_variations_final = variation;
                std::cout << "End of gradient descent reached at iteration ";
                std::cout << n_variations_final << " of " << n_variations << ".";
                std::cout << " Current alpha: " << alphas(variation) << ", ";
                std::cout << "next alpha: " << alphas(variation + 1);
                std::cout << std::endl;
                break;
            }
        }
    }
}
