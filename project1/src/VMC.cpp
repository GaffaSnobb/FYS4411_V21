#include "VMC.h"


void VMC::set_local_energy()
{   /* */
    std::cout << "VMC.cpp: set_local_energy()" << std::endl;
    if (n_dims == 1)
    {
        local_energy_ptr = &local_energy_1d;
    }
    else if (n_dims == 2)
    {
        local_energy_ptr = &local_energy_2d;
    }
    else if (n_dims == 3)
    {
        local_energy_ptr = &local_energy_3d;
    }
}

void VMC::set_wave_function()
{   /* */
    std::cout << "VMC.cpp: set_wave_function()" << std::endl;
    if (n_dims == 1)
    {
        wave_function_exponent_ptr = &wave_function_exponent_1d;
    }
    else if (n_dims == 2)
    {
        wave_function_exponent_ptr = &wave_function_exponent_2d;
    }
    else if (n_dims == 3)
    {
        wave_function_exponent_ptr = &wave_function_exponent_3d;
    }
}

void VMC::set_initial_positions(int dim, int particle)
{   /* set the initial positions */

    //std::cout << "VMC.cpp: set_initial_positions()" << std::endl;

    if (method == "brute_force"){
        pos_current(dim, particle) = step_size*(uniform(engine) - 0.5);
    }
    else if (method == "importance_sampling"){
        pos_current(dim, particle) = normal(engine)*sqrt(time_step);
    }
    else {
        std::cout << "No method chosen"<< std::endl;
    }
}

void VMC::set_new_positions(int dim, int particle)
{   /* fubar */
    //std::cout << "VMC.cpp: set_new_positions()" << std::endl;

    if (method == "brute_force")
    {
        pos_new(dim, particle) = pos_current(dim, particle) + step_size*(uniform(engine) - 0.5);
    }
    else if (method == "importance_sampling")
    {
        pos_new(dim, particle) = pos_current(dim, particle) +
            diffusion_coeff*qforce_current(dim, particle)*time_step +
            normal(engine)*sqrt(time_step);
    }
    else
    {
        std::cout << "No method chosen"<< std::endl;
    }
}

void VMC::brute_force()
{   /*
    Brute-force Monte Carlo simulation using Metropolis.
    */
    // Declared outside loop due to parallelization.
    int particle;       // Index for particle loop.
    int particle_inner; // Index for inner particle loops.
    int _;              // Index for MC loop.
    int dim;            // Index for dimension loops.
    // int brute_force_counter = 0; // Debug counter for the Metropolis algorithm.

    for (int variation = 0; variation < n_variations; variation++)
    {   /*
        Run over all variations.
        */
        e_expectation_squared = 0;
        wave_current = 0;
        for (particle = 0; particle < n_particles; particle++)
        {   /*
            Iterate over all particles.  In this loop, all current
            positions are calulated along with the current wave
            functions.
            */
            for (dim = 0; dim < n_dims; dim++)
            {
                pos_current(dim, particle) = step_size*(uniform(engine) - 0.5);
            }
            wave_current += wave_function_exponent_ptr(
                    pos_current.col(particle),  // Particle position.
                    alphas(variation),
                    beta
                );
        }

        for (_ = 0; _ < n_mc_cycles; _++)
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
                {
                    pos_new(dim, particle) =
                        pos_current(dim, particle) + step_size*(uniform(engine) - 0.5);
                }
                
                wave_new = 0;   // Overwrite the new wave func from previous particle step.
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {   /*
                    After moving one particle, the wave function is
                    calculated based on all particle positions.
                    */
                    wave_new += wave_function_exponent_ptr(
                            pos_new.col(particle_inner),  // Particle position.
                            alphas(variation),
                            beta
                        );
                }

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

                local_energy = 0;   // Overwrite local energy from previous particle step.
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {   /*
                    After moving one particle, the local energy is
                    calculated based on all particle positions.
                    */
                    local_energy += local_energy_ptr(
                        pos_current.col(particle_inner),
                        alphas(variation),
                        beta
                    );
                }

                e_expectations(variation) += local_energy;
                e_expectation_squared += local_energy*local_energy;
            }
        }

        e_expectations(variation) /= n_mc_cycles;
        e_expectation_squared /= n_mc_cycles;
        e_variances(variation) =
            e_expectation_squared - e_expectations(variation)*e_expectations(variation);
    }
// std::cout << "\nbrute_force: " << brute_force_counter/n_mc_cycles << std::endl;
}

void VMC::generalization()
{   /*
    Generalization.
    */
    // Declared outside loop due to parallelization.
    int particle;       // Index for particle loop.
    int particle_inner; // Index for inner particle loops.
    int _;              // Index for MC loop.
    int dim;            // Index for dimension loops.
    // int brute_force_counter = 0; // Debug counter for the Metropolis algorithm.
    double wave_derivative; // Derivative of wave function wrt. alpha.
    double wave_derivative_expectation;    // Wave func expectation value.

    for (int variation = 0; variation < n_variations; variation++)
    {   /*
        Run over all variations.
        */
        e_expectation_squared = 0;
        wave_current = 0;
        for (particle = 0; particle < n_particles; particle++)
        {   /*
            Iterate over all particles.  In this loop, all current
            positions are calulated along with the current wave
            functions.
            */
            for (dim = 0; dim < n_dims; dim++)
            {
                pos_current(dim, particle) = step_size*(uniform(engine) - 0.5);
            }
            wave_current += wave_function_exponent_ptr(
                    pos_current.col(particle),  // Particle position.
                    alphas(variation),
                    beta
                );
        }

        for (_ = 0; _ < n_mc_cycles; _++)
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
                {
                    pos_new(dim, particle) =
                        pos_current(dim, particle) + step_size*(uniform(engine) - 0.5);
                }
                
                wave_new = 0;   // Overwrite the new wave func from previous particle step.
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {   /*
                    After moving one particle, the wave function is
                    calculated based on all particle positions.
                    */
                    wave_new += wave_function_exponent_ptr(
                            pos_new.col(particle_inner),  // Particle position.
                            alphas(variation),
                            beta
                        );
                }

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

                local_energy = 0;   // Overwrite local energy from previous particle step.
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {   /*
                    After moving one particle, the local energy is
                    calculated based on all particle positions.
                    */
                    local_energy += local_energy_ptr(
                        pos_current.col(particle_inner),
                        alphas(variation),
                        beta
                    );
                }

                e_expectations(variation) += local_energy;
                e_expectation_squared += local_energy*local_energy;
            }
        }

        e_expectations(variation) /= n_mc_cycles;
        e_expectation_squared /= n_mc_cycles;
        e_variances(variation) =
            e_expectation_squared - e_expectations(variation)*e_expectations(variation);
    }
// std::cout << "\nbrute_force: " << brute_force_counter/n_mc_cycles << std::endl;
}

void VMC::importance_sampling(double t)
{   /*
    Task 1c importance sampling is implemented here.
    */
    time_step = t;

    //std::cout<<time_step<<std::endl;

    // Declared outside loop due to parallelization.
    int particle;       // Index for particle loop.
    int particle_inner; // Index for inner particle loops.
    int _;              // Index for MC loop.
    int dim;            // Index for dimension loop.

    // int importance_counter = 0; // Debug.

    double wave_derivative; // Derivative of wave function wrt. alpha.
    double wave_derivative_expectation;    // Wave func expectation value.

    for (int variation = 0; variation < n_variations; variation++)
    {   /*
        Run over all variations.
        */
        e_expectation_squared = 0;
        wave_current = 0;
        for (particle = 0; particle < n_particles; particle++)
        {   /*
            Iterate over all particles. Set the initial current positions
            calculate the wave function and quantum force.
            */
            for (dim = 0; dim < n_dims; dim++)
            {
                pos_current(dim, particle) =
                    normal(engine)*sqrt(time_step);

                qforce_current(dim, particle) =
                    -4*alphas(variation)*pos_current(dim, particle);
            }
            wave_current += wave_function_exponent_ptr(
                pos_current.col(particle),  // Particle position.
                alphas(variation),
                beta
            );
        }

        for (_ = 0; _ < n_mc_cycles; _++)
        {   /* Run over all Monte Carlo cycles. */

            for (particle = 0; particle < n_particles; particle++)
            {   /*
                Iterate over all particles. Suggest new positions,
                calculate new wave function and quantum force.
                */
                for (dim = 0; dim < n_dims; dim++)
                {
                    pos_new(dim, particle) = pos_current(dim, particle) +
                        diffusion_coeff*qforce_current(dim, particle)*time_step +
                        normal(engine)*sqrt(time_step);

                    qforce_new(dim, particle) =
                        -4*alphas(variation)*pos_new(dim, particle);
                }
                
                wave_new = 0;   // Overwrite the new wave func from previous particle step.
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {
                    wave_new += wave_function_exponent_ptr(
                        pos_new.col(particle_inner),  // Particle position.
                        alphas(variation),
                        beta
                    );
                }

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
                        pos_current(dim, particle) =
                            pos_new(dim, particle);
                        qforce_current(dim, particle) =
                            qforce_new(dim, particle);
                    }

                    wave_current = wave_new;
                    // importance_counter += 1;    // Debug.
                }
                
                local_energy = 0;   // Overwrite local energy from previous particle step.
                wave_derivative = 0;
                for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
                {
                    local_energy += local_energy_ptr(
                        pos_current.col(particle_inner),
                        alphas(variation),
                        beta
                    );
                    wave_derivative += wave_function_3d_diff_wrt_alpha(
                        pos_current.col(particle_inner),
                        alphas(variation),
                        beta
                    );
                }
                wave_derivative_expectation += wave_derivative;
                e_expectations(variation) += local_energy;
                e_expectation_squared += local_energy*local_energy;
            }
        }

        e_expectations(variation) /= n_mc_cycles;
        e_expectation_squared /= n_mc_cycles;
        e_variances(variation) = e_expectation_squared
            - e_expectations(variation)*e_expectations(variation);

    }
    // std::cout << "\nimportance_sampling: " << importance_counter/n_mc_cycles << std::endl;
}


void VMC::importance_sampling_with_gradient_descent(
    double time_step_input,
    double alpha,
    double &energy_expectation,
    double &energy_derivative
)
{   /*
    Task 1d gradient descent.

    Parameters
    ----------
    time_step_input : double
        Input time step for Greens function (Greens ratio).

    alpha : double
        Variational parameter.

    energy_expectation : double reference
        Reference to energy expectation value.

    energy_derivative : double reference
        Reference to storage for the energy derivative value.
    */
    time_step = time_step_input;

    // Declared outside loop due to parallelization.
    int particle;   // Index for particle loop.
    int _;          // Index for MC loop.
    int dim;        // Index for dimension loop.
    int particle_inner; // Index for inner particle loops.

    // int importance_counter = 0; // Debug.

    double wave_derivative = 0;     // Derivative of wave function wrt. alpha.
    double wave_derivative_expectation = 0;    // Expectation value of the wave function derivative.
    double wave_times_energy_expectation = 0;
    wave_current = 0;   // Reset wave function for each variation.


    for (particle = 0; particle < n_particles; particle++)
    {   /*
        Iterate over all particles. Set the initial current positions
        calculate the wave function and quantum force.
        */
        for (dim = 0; dim < n_dims; dim++)
        {
            pos_current(dim, particle) = normal(engine)*sqrt(time_step);

            qforce_current(dim, particle) =
                -4*alpha*pos_current(dim, particle);
        }
        wave_current += wave_function_exponent_ptr(
            pos_current.col(particle),  // Particle position.
            alpha,
            beta
        );
    }

    for (_ = 0; _ < n_mc_cycles; _++)
    {   /* Run over all Monte Carlo cycles. */

        for (particle = 0; particle < n_particles; particle++)
        {   /*
            Iterate over all particles. Suggest new positions,
            calculate new wave function and quantum force.
            TODO: break lines on long expressions.
            */
            for (dim = 0; dim < n_dims; dim++)
            {
                pos_new(dim, particle) = pos_current(dim, particle) +
                    diffusion_coeff*qforce_current(dim, particle)*time_step +
                    normal(engine)*sqrt(time_step);

                qforce_new(dim, particle) =
                    -4*alpha*pos_new(dim, particle);
            }
            wave_new = 0;   // Overwrite the new wave func from previous particle step.
            for (particle_inner = 0; particle_inner < n_particles; particle_inner++)
            {
                wave_new += wave_function_exponent_ptr(
                    pos_new.col(particle_inner),  // Particle position.
                    alpha,
                    beta
                );
            }

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
                    pos_current(dim, particle) =
                        pos_new(dim, particle);
                    qforce_current(dim, particle) =
                        qforce_new(dim, particle);
                }

                wave_current = wave_new;
                // importance_counter += 1;    // Debug.
            }
            
            local_energy = 0;   // Overwrite local energy from previous particle step.
            wave_derivative = 0;
            for (particle_inner = 0; particle_inner < n_particles; particle_inner++)    // I believe this can be put into the Metropolis if.
            {
                local_energy += local_energy_ptr(
                    pos_current.col(particle_inner),
                    alpha,
                    beta
                );
                wave_derivative += wave_function_3d_diff_wrt_alpha(
                    pos_current.col(particle_inner),
                    alpha,
                    beta
                );
            }
            wave_derivative_expectation += wave_derivative;
            energy_expectation += local_energy;
            wave_times_energy_expectation += wave_derivative*local_energy;
        }
    }
    wave_times_energy_expectation /= n_mc_cycles;
    wave_derivative_expectation /= n_mc_cycles;
    energy_expectation /= n_mc_cycles;
    energy_derivative = 2*(wave_times_energy_expectation - wave_derivative_expectation*energy_expectation/10);
    std::cout << "energy_expectation: " << energy_expectation << std::endl;
    std::cout << "wave_derivative_expectation: " << wave_derivative_expectation << std::endl;
    std::cout << "wave_derivative_expectation*energy_expectation: " << wave_derivative_expectation*energy_expectation << std::endl;
    std::cout << "wave_times_energy_expectation: " << wave_times_energy_expectation << std::endl;
    std::cout << "energy_derivative: " << energy_derivative << std::endl;
    std::cout << "wave_derivative: " << wave_derivative << std::endl;
    // std::cout << "\n";
}



void VMC::write_to_file(std::string fpath)
{
    outfile.open(fpath, std::ios::out);
    outfile << std::setw(20) << "alpha";
    outfile << std::setw(20) << "variance_energy";
    outfile << std::setw(21) << "expected_energy\n";

    for (int i = 0; i < n_variations; i++)
    {   /*
        Write data to file.
        */
        outfile << std::setw(20) << std::setprecision(10);
        outfile << alphas(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_variances(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_expectations(i) << "\n";
    }
    outfile.close();
}
