#include "VMC.h"

VMC::VMC(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input,
    arma::Col<double> alphas_input,
    const double beta_input,
    bool debug_input
) : n_dims(n_dims_input),
    n_variations(n_variations_input),
    n_mc_cycles(n_mc_cycles_input),
    n_particles(n_particles_input),
    beta(beta_input)

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

    debug_input : boolean
        For toggling debug print on / off.
    */
    pos_new = arma::Mat<double>(n_dims, n_particles);         // Proposed new position.
    pos_current = arma::Mat<double>(n_dims, n_particles);     // Current position.
    e_variances = arma::Col<double>(n_variations);            // Energy variances.
    e_expectations = arma::Col<double>(n_variations);         // Energy expectation values.
    qforce_current = arma::Mat<double>(n_dims, n_particles);  // Current quantum force.
    qforce_new = arma::Mat<double>(n_dims, n_particles);      // New quantum force.
    test_local = arma::Row<double>(n_mc_cycles);              // Temporary
    energies = arma::Mat<double>(n_mc_cycles, n_variations);
    alphas = alphas_input;
    n_variations_final = n_variations;  // If stop condition is not reached.
    debug = debug_input;                // For toggling debug print on / off.

    acceptances = arma::Col<double>(n_variations);   // Debug.
    acceptances.zeros();

    e_expectations.zeros(); // Array must be zeroed since values will be added to it.
    energies.zeros();
    engine.seed(seed);


    // set_local_energy();  // Moved to main.cpp.
    // set_wave_function(); // Moved to main.cpp.
}

void VMC::set_seed(double seed_input)
{   
    seed = seed_input;
    engine.seed(seed);
}

void VMC::set_quantum_force(bool interaction)
{
    if ((n_dims == 1) and !(interaction))
    {
        not_implemented_error("quantum force", interaction);
    }
    else if ((n_dims == 2) and !(interaction))
    {
        not_implemented_error("quantum force", interaction);
    }
    else if ((n_dims == 3) and !(interaction))
    {
        quantum_force_ptr = &quantum_force_3d_no_interaction;
    }
    else if ((n_dims == 1) and interaction)
    {
        not_implemented_error("quantum force", interaction);
    }
    else if ((n_dims == 2) and interaction)
    {
        not_implemented_error("quantum force", interaction);
    }
    else if ((n_dims == 3) and interaction)
    {   
        quantum_force_ptr = &quantum_force_3d_interaction;
    }
    else
    {
        not_implemented_error("quantum force", interaction);
    }
    call_set_quantum_force = true;
}

void VMC::set_local_energy(bool interaction)
{   /*
    Set pointers to the correct local energy function.

    Parameters
    ----------
    interaction : boolean
        Toggle interaction between particles on / off.
    */

    if ((n_dims == 1) and !(interaction))
    {
        local_energy_ptr = &local_energy_1d_no_interaction;
    }
    else if ((n_dims == 2) and !(interaction))
    {
        local_energy_ptr = &local_energy_2d_no_interaction;
    }
    else if ((n_dims == 3) and !(interaction))
    {
        local_energy_ptr = &local_energy_3d_no_interaction;
    }
    else if ((n_dims == 1) and (interaction))
    {
        not_implemented_error("local energy", interaction);
    }
    else if ((n_dims == 2) and (interaction))
    {
        not_implemented_error("local energy", interaction);
    }
    else if ((n_dims == 3) and (interaction))
    {    
        local_energy_ptr = &local_energy_3d_interaction;        
    }

    call_set_local_energy = true;
}

void VMC::set_wave_function(bool interaction)
{   /*
    Set pointers to the correct wave function exponent.

    Parameters
    ----------
    interaction : boolean
        Toggle interaction between particles on / off.
    */

    if ((n_dims == 1) and !(interaction))
    {
        wave_function_exponent_ptr = &wave_function_exponent_1d_no_interaction;
    }
    else if ((n_dims == 2) and !(interaction))
    {
        wave_function_exponent_ptr = &wave_function_exponent_2d_no_interaction;
    }
    else if ((n_dims == 3) and !(interaction))
    {
        wave_function_ptr = &wave_function_3d_no_interaction_with_loop;
        wave_function_exponent_ptr = &wave_function_exponent_3d_no_interaction;
    }
    else if ((n_dims == 1) and (interaction))
    {
        not_implemented_error("wave function", interaction);
    }
    else if ((n_dims == 2) and (interaction))
    {
        not_implemented_error("wave function", interaction);
    }
    else if ((n_dims == 3) and (interaction))
    {
        wave_function_ptr = &wave_function_3d_interaction_with_loop;
    }

    call_set_wave_function = true;
}

void VMC::not_implemented_error(std::string name, bool interaction)
{   
    std::cout << "NotImplementedError" << std::endl;
    std::cout << "Cannot set " << name << " for dimensions: " << n_dims;
    std::cout << " and interaction: " << interaction << "." << std::endl;
    exit(0);
}

void VMC::one_variation(int variation)
{
    std::cout << "NotImplementedError" << std::endl;
}

void VMC::solve()
{   /*
    Iterate over variational parameters. Extract energy variances and
    expectation values.
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

    #ifdef _OPENMP
        double t1;
        double t2;
        double comp_time;
    #else
        std::chrono::steady_clock::time_point t1;
        std::chrono::steady_clock::time_point t2;
        std::chrono::duration<double> comp_time;
    #endif

    for (int variation = 0; variation < n_variations; variation++)
    {
        #ifdef _OPENMP
            t1 = omp_get_wtime();
        #else
            t1 = std::chrono::steady_clock::now();
        #endif

        one_variation(variation);
        e_expectations(variation) = energy_expectation;
        e_variances(variation) = energy_variance;

        std::cout << "variation : " << std::setw(3) <<  variation;
        std::cout << ", alpha: " << std::setw(10) << alphas(variation);
        std::cout << ", energy: " << energy_expectation;
        std::cout << ", acceptance: " << std::setw(7) << acceptances(variation)/(n_mc_cycles*n_particles);
        
        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
            std::cout << ",  time : " << comp_time << "s" << std::endl;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            std::cout << ",  time : " << comp_time.count() << "s" << std::endl;
        #endif
        
    }
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

void VMC::write_to_file_particles(std::string fpath)
{
    outfile.open(fpath, std::ios::out);
    outfile << std::setw(20) << "alpha";
    outfile << std::setw(20) << "variance_energy";
    outfile << std::setw(21) << "expected_energy\n";

    for (int i = 0; i < n_variations_final; i++)
    {   /*
        Write data to file.
        */
        outfile << std::setw(20) << std::setprecision(10);
        outfile << alphas(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_variances(i)/n_particles;
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_expectations(i)/n_particles << "\n";
    }
    outfile.close();
}

void VMC::write_energies_to_file(std::string fpath)
{
    outfile.open(fpath, std::ios::out);

    for (int i = 0; i < n_variations; i++){
      outfile << std::setw(20) << std::setprecision(10);
      outfile << alphas(i);
    }

    outfile << "\n";
    energies.save(outfile, arma::raw_ascii);
    outfile.close();
}

VMC::~VMC()
{
    // acceptances.print();
}