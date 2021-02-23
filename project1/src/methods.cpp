#include "methods.h"

void BruteForce::set_initial_positions(int dim, int particle, int variation)
{   /*
    Set the initial positions according to the brute force approach,
    drawing random numbers uniformly in the interval [-1/2, 1/2].

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    variation : integer
        Current variation index.
    */
    pos_current(dim, particle) = step_size*(uniform(engine) - 0.5);
}

void BruteForce::set_new_positions(int dim, int particle, int variation)
{   /*
    Set new positions according to the brute force approach, drawing
    random numbers uniformly in the interval [-1/2, 1/2].

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    variation : integer
        Current variation index.
    */
    pos_new(dim, particle) = pos_current(dim, particle) + step_size*(uniform(engine) - 0.5);
}

void BruteForce::metropolis(int dim, int particle)
{
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
        // brute_force_counter += 1;   // Debug.
    }
}

void ImportanceSampling::set_initial_positions(int dim, int particle, int variation)
{   /*
    Set the initial positions according to the importance sampling
    approach, drawing random numbers normally distributed.

    TODO: Consider removing dim, particle, variation as function
    arguments since they are class variables.

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    variation : integer
        Current variation index.
    */
    pos_current(dim, particle) = normal(engine)*sqrt(time_step);
    
    qforce_current(dim, particle) = -4*alphas(variation)*pos_current(dim, particle);
}

void ImportanceSampling::set_new_positions(int dim, int particle, int variation)
{   /*
    Set new positions according to the importance sampling approach,
    drawing random numbers normally distributed.

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.

    variation : integer
        Current variation index.
    */
    pos_new(dim, particle) = pos_current(dim, particle) +
        diffusion_coeff*qforce_current(dim, particle)*time_step +
        normal(engine)*sqrt(time_step);
    
    qforce_new(dim, particle) = -4*alphas(variation)*pos_new(dim, particle);
}

void ImportanceSampling::metropolis(int dim, int particle)
{
    double greens_ratio = 0.0;
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
    }
}