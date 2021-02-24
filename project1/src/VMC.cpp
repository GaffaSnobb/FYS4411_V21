#include "VMC.h"

VMC::VMC(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input
) : n_dims(n_dims_input),
    n_variations(n_variations_input),
    n_mc_cycles(n_mc_cycles_input),
    n_particles(n_particles_input)

{   /*
    Class constructor.

    Parameters
    ----------
    n_dims_input : constant integer
        The number of spatial dimensions.
    */

    // Pre-filling the alphas vector due to parallelization.
    alphas.fill(alpha_step);
    alphas = arma::cumsum(alphas);
    e_expectations.zeros(); // Array must be zeroed since values will be added.
    engine.seed(seed);

    set_local_energy();
    set_wave_function();
}

void VMC::set_local_energy()
{   /*
    Set pointers to the correct local energy function.
    */
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
{   /*
    Set pointers to the correct wave function exponent.
    */
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

void VMC::set_initial_positions(int dim, int particle, double alpha)
{   /*
    This function will be overwritten by child class method.
    */
    std::cout << "NotImplementedError" << std::endl;
}

void VMC::set_new_positions(int dim, int particle, double alpha)
{   /*
    This function will be overwritten by child class method.
    */
    std::cout << "NotImplementedError" << std::endl;
}

void VMC::metropolis(int dim, int particle, double alpha)
{   /*
    This function will be overwritten by child class method.
    */
    std::cout << "NotImplementedError" << std::endl;
}

void VMC::solve()
{   /*
    This function will be overwritten by child class method.  The solve
    method handles the progression of the variational parameter, alpha.
    */
    std::cout << "NotImplementedError" << std::endl;
}

void VMC::one_variation(double alpha)
{   /*
    Perform calculations for a single variational parameter.

    Parameters
    ----------
    alpha : double
        Variational parameter.
    */
    wave_current = 0;   // Reset wave function for each variation.
    energy_expectation = 0; // Reset for each variation.
    energy_variance = 0; // Reset for each variation.

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
            set_initial_positions(dim, particle, alpha);
        }
        wave_current += wave_function_exponent_ptr(
            pos_current.col(particle),  // Particle position.
            alpha,
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
                set_new_positions(dim, particle, alpha);
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

            metropolis(dim, particle, alpha);

            energy_expectation += local_energy;
            e_expectation_squared += local_energy*local_energy;
        }
    }

    energy_expectation /= n_mc_cycles;
    e_expectation_squared /= n_mc_cycles;
    energy_variance = e_expectation_squared
        - energy_expectation*energy_expectation;
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
