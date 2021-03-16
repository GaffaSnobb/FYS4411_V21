#include "methods.h"

BruteForce::BruteForce(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input,
    arma::Col<double> alphas_input,
    const double brute_force_step_size_input,
    bool debug_input
) : VMC(
        n_dims_input,
        n_variations_input,
        n_mc_cycles_input,
        n_particles_input,
        alphas_input,
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

    wave_current = 0;       // Reset wave function for each variation.
    energy_expectation = 0; // Reset for each variation.
    energy_variance = 0;    // Reset for each variation. NB: Variable not inside parallel region.
    energy_expectation_squared = 0;

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
        wave_current += wave_function_exponent_ptr(
            pos_current.col(particle),  // Position of one particle.
            alpha,
            beta
        );
    }
    
    #pragma omp parallel for \
        private(mc, particle, dim, particle_inner) \
        private(wave_new, exponential_diff) \
        firstprivate(wave_current, local_energy) \
        firstprivate(pos_new, pos_current) \
        reduction(+:acceptance, energy_expectation, energy_expectation_squared)
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

            wave_new = 0;   // Overwrite the new wave func from previous particle step.
            for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
            {   /*
                After moving one particle, the wave function is
                calculated based on all particle positions.
                */
                wave_new += wave_function_exponent_ptr(
                        pos_new.col(particle_inner),  // Particle position.
                        alpha,
                        beta
                    );
            }

            exponential_diff = 2*(wave_new - wave_current);

            if (uniform(engine) < std::exp(exponential_diff))
            {   /*
                Perform the Metropolis algorithm.  To save one
                exponential calculation, the difference is taken
                of the exponents instead of the ratio of the
                exponentials. Marginally better...
                */
                acceptance++;    // Debug.
                for (dim = 0; dim < n_dims; dim++)
                {
                    pos_current(dim, particle) = pos_new(dim, particle);
                }
                wave_current = wave_new;

                local_energy = 0;   // Overwrite local energy from previous particle step.
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {   /*
                    After moving one particle, the local energy is
                    calculated based on all particle positions.
                    */
                    local_energy += local_energy_ptr(
                        pos_current.col(particle_inner),
                        alpha,
                        beta
                    );
                }
            }

            energy_expectation += local_energy;
            energy_expectation_squared += local_energy*local_energy;
        }
        energies(mc, variation) = local_energy;
    }

    energy_expectation /= n_mc_cycles;
    energy_expectation_squared /= n_mc_cycles;
    energy_variance = energy_expectation_squared
        - energy_expectation*energy_expectation/n_particles;

    acceptances(variation) = acceptance;    // Debug.

    //std::cout << "alpha:    " << alpha  << std::endl;
    //std::cout << "<E^2>:    " << energy_expectation_squared <<std::endl;
    //std::cout << "<E>^2:    " << energy_expectation*energy_expectation << std::endl;
    //std::cout << "sigma^2:  " << energy_variance << std::endl;
    //std::cout << "" << std::endl;

}

ImportanceSampling::ImportanceSampling(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input,
    arma::Col<double> alphas_input,
    const double importance_time_step_input,
    bool debug_input
) : VMC(
        n_dims_input,
        n_variations_input,
        n_mc_cycles_input,
        n_particles_input,
        alphas_input,
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
    wave_current = 0;
    energy_expectation = 0;
    energy_variance = 0;
    energy_expectation_squared = 0;

    // GD specifics.
    wave_derivative = 0;
    wave_derivative_expectation = 0;
    wave_times_energy_expectation = 0;
    // GD specifics end.

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
            // qforce_current(dim, particle) = -4*alpha*pos_current(dim, particle);
        }

        qforce_current.col(particle) =
            quantum_force_ptr(pos_current.col(particle), alpha);
        wave_current += wave_function_exponent_ptr(
            pos_current.col(particle),  // Position of one particle.
            alpha,
            beta
        );
    }
    
    #pragma omp parallel for \
        private(mc, particle, dim, particle_inner) \
        private(wave_new, exponential_diff) \
        firstprivate(wave_current, local_energy) \
        firstprivate(pos_new, qforce_new, pos_current, qforce_current) \
        reduction(+:acceptance, energy_expectation, energy_expectation_squared) \
        reduction(+:wave_times_energy_expectation, wave_derivative_expectation) \
        firstprivate(wave_derivative)
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
                pos_new(dim, particle) = pos_current(dim, particle) +
                    diffusion_coeff*qforce_current(dim, particle)*time_step +
                    normal(engine)*sqrt(time_step);
                
                // qforce_new(dim, particle) = -4*alpha*pos_new(dim, particle);
            }

            qforce_new.col(particle) = quantum_force_ptr(pos_new.col(particle), alpha);

            wave_new = 0;   // Overwrite the new wave func from previous particle step.
            for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
            {   /*
                After moving one particle, the wave function is
                calculated based on all particle positions.
                */
                wave_new += wave_function_exponent_ptr(
                        pos_new.col(particle_inner),  // Particle position.
                        alpha,
                        beta
                    );
            }

            double greens_ratio = 0;
            for (int dim = 0; dim < n_dims; dim++)
            {   /*
                Calculate greens ratio for the acceptance
                criterion.
                */
                greens_ratio +=
                    0.5*(qforce_current(dim, particle) + qforce_new(dim, particle))
                    *(0.5*diffusion_coeff*time_step*
                    (qforce_current(dim, particle) - qforce_new(dim, particle))
                    - pos_new(dim, particle) + pos_current(dim, particle));
            }

            greens_ratio = exp(greens_ratio);
            exponential_diff = 2*(wave_new - wave_current);

            if (uniform(engine) < greens_ratio*std::exp(exponential_diff))
            {   /*
                Metropolis step with new acceptance criterion.
                */
                acceptance++;    // Debug.
                for (dim = 0; dim < n_dims; dim++)
                {   /*
                    TODO: Can prob. drop this loop by
                    pos_current.col(particle) = pos_new.col(particle).
                    */
                    pos_current(dim, particle) = pos_new(dim, particle);
                    qforce_current(dim, particle) = qforce_new(dim, particle);
                }
                wave_current = wave_new;

                local_energy = 0;   // Overwrite local energy from previous particle step.
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {   /*
                    After moving one particle, the local energy is
                    calculated based on all particle positions.
                    */
                    local_energy += local_energy_ptr(
                        pos_current.col(particle_inner),
                        alpha,
                        beta
                    );
                }

                wave_derivative = 0;
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {   /*
                    Calculations needed for gradient descent.
                    */
                    wave_derivative += wave_function_3d_diff_wrt_alpha(
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
            
            energy_expectation += local_energy;
            energy_expectation_squared += local_energy*local_energy;
        }
        energies(mc, variation) = local_energy;
    }

    energy_expectation /= n_mc_cycles;
    energy_expectation_squared /= n_mc_cycles;
    energy_variance = energy_expectation_squared
        - energy_expectation*energy_expectation/n_particles;

    // GD specifics.
    wave_times_energy_expectation /= n_mc_cycles;
    wave_derivative_expectation /= n_mc_cycles;
    // GD specifics end.

    acceptances(variation) = acceptance;    // Debug.
}

GradientDescent::GradientDescent(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input,
    const double importance_time_step_input,
    const double learning_rate_input,
    const double initial_alpha_input,
    bool debug_input
) : ImportanceSampling(
        n_dims_input,
        n_variations_input,
        n_mc_cycles_input,
        n_particles_input,
        arma::linspace(0, 0, n_variations_input),   // Dummy input. Not in use.
        importance_time_step_input,
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

void GradientDescent::solve()
{   /*
    Iterate over variational parameters.  Use gradient descent to
    efficiently calculate alphas.
    */
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
            wave_derivative_expectation*energy_expectation/n_particles);
        
        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time_chrono = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            comp_time = comp_time_chrono.count();
        #endif

        alphas(variation + 1) = alphas(variation) - learning_rate*energy_derivative;

        if (debug)
        {
            std::cout << "variation : " << std::setw(3) <<  variation;
            std::cout << ", alpha: " << std::setw(10) << alphas(variation);
            // std::cout << ", energy: " << energy_expectation;
            std::cout << ", acceptance: " << std::setw(7) << acceptances(variation)/(n_mc_cycles*n_particles);
            std::cout << ",  time : " << comp_time << "s" << std::endl;

            std::cout << "energy_expectation: " << energy_expectation << std::endl;
            std::cout << "wave_derivative_expectation: " << wave_derivative_expectation << std::endl;
            std::cout << "wave_derivative_expectation*energy_expectation/n_particles: " << wave_derivative_expectation*energy_expectation/n_particles << std::endl;
            std::cout << "wave_times_energy_expectation: " << wave_times_energy_expectation << std::endl;
            std::cout << "energy_derivative: " << energy_derivative << std::endl;
            std::cout << "wave_derivative: " << wave_derivative << std::endl;
            std::cout << "\n";
        }

        if ( std::abs(alphas(variation + 1) - alphas(variation)) < 1e-4 )
        {
            n_variations_final = variation;
            std::cout << "End of gradient descent reached at iteration ";
            std::cout << n_variations_final << " of " << n_variations << ".";
            std::cout << std::endl;
            break;
        }
    }
}