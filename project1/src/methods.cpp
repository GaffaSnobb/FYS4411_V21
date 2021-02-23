#include "methods.h"

void BruteForce::set_initial_positions(int dim, int particle, double alpha)
{   /*
    Set the initial positions according to the brute force approach,
    drawing random numbers uniformly in the interval [-1/2, 1/2].

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    alpha : double
        Variational parameter. Unused here.
    */
    pos_current(dim, particle) = step_size*(uniform(engine) - 0.5);
}

void BruteForce::set_new_positions(int dim, int particle, double alpha)
{   /*
    Set new positions according to the brute force approach, drawing
    random numbers uniformly in the interval [-1/2, 1/2].

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    alpha : double
        Variational parameter. Unused here.
    */
    pos_new(dim, particle) = pos_current(dim, particle) + step_size*(uniform(engine) - 0.5);
}

void BruteForce::solve()
{   /*
    Iterate over variational parameters. Extract energy variances and
    expectation values.
    */
    for (int variation = 0; variation < n_variations; variation++)
    {
        one_variation(alphas(variation));
        e_expectations(variation) = energy_expectation;
        e_variances(variation) = energy_variance;
    }
}

void BruteForce::metropolis(int dim, int particle, double alpha)
{   /*
    Metropolis specifics for brute force.
    */
    exponential_diff =
        2*(wave_new - wave_current);

    if (uniform(engine) < std::exp(exponential_diff))
    {   /*
        Perform the Metropolis algorithm.  To save one
        exponential calculation, the difference is taken
        of the exponents instead of the ratio of the
        exponentials. Marginally better...
        */
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
}

void ImportanceSampling::set_initial_positions(int dim, int particle, double alpha)
{   /*
    Set the initial positions according to the importance sampling
    approach, drawing random numbers normally distributed.

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    alpha : double
        Variational parameter.
    */
    pos_current(dim, particle) = normal(engine)*sqrt(time_step);
    
    qforce_current(dim, particle) = -4*alpha*pos_current(dim, particle);
}

void ImportanceSampling::set_new_positions(int dim, int particle, double alpha)
{   /*
    Set new positions according to the importance sampling approach,
    drawing random numbers normally distributed.

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    alpha : double
        Variational parameter.
    */
    pos_new(dim, particle) = pos_current(dim, particle) +
        diffusion_coeff*qforce_current(dim, particle)*time_step +
        normal(engine)*sqrt(time_step);
    
    qforce_new(dim, particle) = -4*alpha*pos_new(dim, particle);
}

void ImportanceSampling::solve()
{   /*
    Iterate over variational parameters. Extract energy variances and
    expectation values.
    */
    for (int variation = 0; variation < n_variations; variation++)
    {
        one_variation(alphas(variation));
        e_expectations(variation) = energy_expectation;
        e_variances(variation) = energy_variance;
    }
}

void ImportanceSampling::metropolis(int dim, int particle, double alpha)
{   /*
    Metropolis specifics for importance sampling.

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    alpha : double
        Variational parameter.
    */
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
        for (dim = 0; dim < n_dims; dim++)
        {
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
    }
}

void GradientDescent::solve()
{   /*
    Iterate over variational parameters.  Use gradient descent to
    efficiently calculate alphas.

    TODO: Add a cut-off for when the proposed new alpha is adequately
    close to the desired value.
    */
    wave_derivative_expectation = 0;
    wave_times_energy_expectation = 0;
    alphas(0) = 0.1; // Initial value.
    double learning_rate = 0.001;
    time_step = 0.05;
    double energy_derivative = 0;
    
    for (int variation = 0; variation < n_variations - 1; variation++)
    {
        one_variation(alphas(variation));
        e_expectations(variation) = energy_expectation;
        e_variances(variation) = energy_variance;
        
        wave_derivative_expectation /= n_mc_cycles;
        wave_times_energy_expectation /= n_mc_cycles;
        energy_derivative = 2*(wave_times_energy_expectation - wave_derivative_expectation*energy_expectation/n_particles);
        
        alphas(variation + 1) = alphas(variation) - learning_rate*energy_derivative;
        
        // std::cout << "energy_expectation: " << energy_expectation << std::endl;
        // std::cout << "wave_derivative_expectation: " << wave_derivative_expectation << std::endl;
        // std::cout << "wave_derivative_expectation*energy_expectation: " << wave_derivative_expectation*energy_expectation << std::endl;
        // std::cout << "wave_times_energy_expectation: " << wave_times_energy_expectation << std::endl;
        // std::cout << "energy_derivative: " << energy_derivative << std::endl;
        // std::cout << "wave_derivative: " << wave_derivative << std::endl;
        // std::cout << "alpha: " << alphas(variation) << std::endl;
        // std::cout << "\n";
    }
}

void GradientDescent::metropolis(int dim, int particle, double alpha)
{   /*
    Metropolis specifics for gradient descent.  In here you find both
    the Metropolis criterion calculations and additional values needed
    to perform gradient descent.

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    alpha : double
        Variational parameter.
    */
    ImportanceSampling::metropolis(dim, particle, alpha);
    
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
    
    wave_derivative_expectation += wave_derivative;
    wave_times_energy_expectation += wave_derivative*local_energy;
}