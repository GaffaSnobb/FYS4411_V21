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

    if (method == 0){
      pos_current(dim, particle) = step_size*(uniform(engine) - 0.5);
    }
    else if (method == 1){
      pos_current(dim, particle) = normal(engine)*sqrt(time_step);
    }
    else {
      std::cout << "No method chosen"<< std::endl;
    }
}

void VMC::set_new_positions(int dim, int particle)
{   /* fubar */
    //std::cout << "VMC.cpp: set_new_positions()" << std::endl;

    if (method == 0)
    {
      pos_new(dim, particle) = pos_current(dim, particle) + step_size*(uniform(engine) - 0.5);
    }
    else if (method == 1)
    {
      pos_new(dim, particle) = pos_current(dim, particle) +
          diffusion_coeff*qforce_current(dim, particle)*time_step +
          normal(engine)*sqrt(time_step);
    }
    else {
      std::cout << "No method chosen"<< std::endl;
    }
}

void VMC::brute_force()
{   /*
    Brute-force Monte Carlo simulation using Metropolis.
    */

    // Declared outside loop due to parallelization.
    int particle;   // Index for particle loop.
    int _;          // Index for MC loop.
    int dim;        // Index for dimension loop.
    int brute_force_counter = 0; // Debug counter for the Metropolis algorithm.

    for (int i = 0; i < n_variations; i++)
    {   /*
        Run over all variations.
        */
        e_expectation_squared = 0;

        for (particle = 0; particle < n_particles; particle++)
        {   /*
            Iterate over all particles.  In this loop, all current
            positions are calulated along with the current wave
            functions.
            */
            for (dim = 0; dim < n_dims; dim++)
            {
                set_initial_positions(dim, particle);
            }
            wave_current(particle) =
                wave_function_exponent_ptr(
                    pos_current.col(particle),  // Particle position.
                    alphas(i),
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
                    set_new_positions(dim, particle);
                }
                wave_new(particle) =
                    wave_function_exponent_ptr(
                        pos_new.col(particle),  // Particle position.
                        alphas(i),
                        beta
                    );

                exponential_diff =
                    2*(wave_new(particle) - wave_current(particle));

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
                    wave_current(particle) = wave_new(particle);
                    brute_force_counter += 1;   // Debug.
                }

                energy_step = local_energy_ptr(
                    pos_current.col(particle),
                    alphas(i),
                    beta
                );

                e_expectations(i) += energy_step;
                e_expectation_squared += energy_step*energy_step;
            }
        }

        e_expectations(i) /= n_mc_cycles;
        e_expectation_squared /= n_mc_cycles;
        e_variances(i) =
            e_expectation_squared - e_expectations(i)*e_expectations(i);
    }
std::cout << "\nbrute_force: " << brute_force_counter/n_mc_cycles << std::endl;
}


void VMC::importance_sampling(double t)
{   /*
    Task 1c importance sampling is implemented here.
    */
    time_step = t;

    // Declared outside loop due to parallelization.
    int particle;   // Index for particle loop.
    int _;          // Index for MC loop.
    int dim;        // Index for dimension loop.
    int importance_counter = 0;

    for (int i = 0; i < n_variations; i++)
    {   /*
        Run over all variations.
        */
        e_expectation_squared = 0;

        for (particle = 0; particle < n_particles; particle++)
        {   /*
            Iterate over all particles. Set the initial current positions
            calculate the wave function and quantum force.
            */
            for (dim = 0; dim < n_dims; dim++)
            {
                set_initial_positions(dim, particle);

                qforce_current(dim, particle) =
                    -4*alphas(i)*pos_current(dim, particle);
            }
            wave_current(particle) =
                wave_function_exponent_ptr(
                    pos_current.col(particle),  // Particle position.
                    alphas(i),
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
                    set_new_positions(dim, particle);

                    qforce_new(dim, particle) =
                        -4*alphas(i)*pos_new(dim, particle);
                }
                wave_new(particle) =
                    wave_function_exponent_ptr(
                        pos_new.col(particle),  // Particle position.
                        alphas(i),
                        beta
                    );

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
                exponential_diff =
                    2*(wave_new(particle) - wave_current(particle));

                if (uniform(engine) < greens_ratio*std::exp(exponential_diff))
                {   /*
                    Metropolis step with new acceptance criterion.
                    */
                    for (dim = 0; dim < n_dims; dim++)
                    {
                        pos_current(dim, particle) = pos_new(dim, particle);
                        qforce_current(dim, particle) = qforce_new(dim, particle);
                    }

                    wave_current(particle) = wave_new(particle);
                    importance_counter += 1;    // Debug.
                }
                energy_step = local_energy_ptr(
                    pos_current.col(particle),
                    alphas(i),
                    beta
                );
                e_expectations(i) += energy_step;
                e_expectation_squared += energy_step*energy_step;
            }
        }

        e_expectations(i) /= n_mc_cycles;
        e_expectation_squared /= n_mc_cycles;
        e_variances(i) =
        e_expectation_squared - e_expectations(i)*e_expectations(i);
    }
    std::cout << "\nimportance_sampling: " << importance_counter/n_mc_cycles << std::endl;
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
