#include "VMC.h"

VMC::VMC(
    const int n_dims_input,
    const int n_variations_input,
    const int n_mc_cycles_input,
    const int n_particles_input,
    arma::Col<double> alphas_input,
    const double beta_input,
    const bool numerical_differentiation_input,
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
    numerical_differentiation = numerical_differentiation_input;
    debug = debug_input;                // For toggling debug print on / off.

    acceptances = arma::Col<double>(n_variations);   // Debug.
    acceptances.zeros();

    e_expectations.zeros(); // Array must be zeroed since values will be added to it.
    energies.zeros();
    engine.seed(seed);

    timing = arma::Col<double>(n_variations);
    timing.zeros();

    // One-body density parameters.
    n_bins = 50;
    r_bins_end = 3;
    bin_locations = arma::linspace(0, r_bins_end - r_bins_end/n_bins, n_bins + 1);
    particle_per_bin_count = arma::Mat<double>(n_bins, n_variations);
    particle_per_bin_count.zeros();
    particle_per_bin_count_thread = arma::Col<double>(n_bins);
    particle_per_bin_count_thread.zeros();
    // One-body density parameters end.
}

void VMC::set_seed(double seed_input)
{
    seed = seed_input;
    engine.seed(seed);
}

void VMC::set_quantum_force(bool interaction)
{
    if ((n_dims == 1) and !interaction)
    {
        quantum_force_ptr = &quantum_force_3d_no_interaction;   // NB!! This might be wrong!
    }
    else if ((n_dims == 2) and !interaction)
    {
        quantum_force_ptr = &quantum_force_3d_no_interaction;   // NB!! This might be wrong!
    }
    else if ((n_dims == 3) and !interaction)
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

    if ((n_dims == 1) and !interaction and !numerical_differentiation)
    {
        local_energy_ptr = &local_energy_1d_no_interaction;
    }
    else if ((n_dims == 1) and !interaction and numerical_differentiation)
    {
        local_energy_ptr = &local_energy_1d_no_interaction_numerical_differentiation;
    }
    else if ((n_dims == 1) and interaction and !numerical_differentiation)
    {
        not_implemented_error("local energy", interaction);
    }
    else if ((n_dims == 2) and !interaction and !numerical_differentiation)
    {
        local_energy_ptr = &local_energy_2d_no_interaction;
    }
    else if ((n_dims == 2) and !interaction and numerical_differentiation)
    {
        local_energy_ptr = &local_energy_2d_no_interaction_numerical_differentiation;
    }
    else if ((n_dims == 2) and interaction and !numerical_differentiation)
    {
        not_implemented_error("local energy", interaction);
    }
    else if ((n_dims == 3) and !interaction and !numerical_differentiation)
    {
        local_energy_ptr = &local_energy_3d_no_interaction;
    }
    else if ((n_dims == 3) and interaction and !numerical_differentiation)
    {
        local_energy_ptr = &local_energy_3d_interaction;
        // local_energy_ptr = &local_energy_3d_interaction_vala;
    }
    else if ((n_dims == 3) and !interaction and numerical_differentiation)
    {
        local_energy_ptr = &local_energy_3d_no_interaction_numerical_differentiation;
    }
    else
    {
        not_implemented_error("local energy", interaction);
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
        wave_function_ptr = &wave_function_1d_no_interaction_with_loop;
        wave_function_diff_wrt_alpha_ptr = &wave_function_1d_diff_wrt_alpha;
    }
    else if ((n_dims == 2) and !(interaction))
    {
        wave_function_ptr = &wave_function_2d_no_interaction_with_loop;
        wave_function_diff_wrt_alpha_ptr = &wave_function_2d_diff_wrt_alpha;
    }
    else if ((n_dims == 3) and !(interaction))
    {
        wave_function_ptr = &wave_function_3d_no_interaction_with_loop;
        wave_function_diff_wrt_alpha_ptr = &wave_function_3d_diff_wrt_alpha;
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
        wave_function_diff_wrt_alpha_ptr = &wave_function_3d_diff_wrt_alpha;
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
        std::cout << ", energy: " << std::setw(10) << energy_expectation;
        std::cout << ", variance: " << std::setw(10) << energy_variance;
        std::cout << ", acceptance: " << std::setw(10) << acceptances(variation)/(n_mc_cycles*n_particles);

        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
            std::cout << ",  time : " << comp_time << "s" << std::endl;
            timing(variation) = comp_time;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            std::cout << ",  time : " << comp_time.count() << "s" << std::endl;
            timing(variation) = comp_time.count();
        #endif

    }
}

void VMC::write_to_file(std::string fpath)
{   /*
    Write data to file. Columns 1, 2, 3 are: alpha, energy variance,
    energy expectation value.

    Parameters
    ----------
    fpath : std::string
        Relative file path and name.
    */
    outfile.open(fpath, std::ios::out);
    outfile << std::setw(20) << "alpha";
    outfile << std::setw(20) << "variance_energy";
    outfile << std::setw(21) << "expected_energy";
    outfile << std::setw(21) << "time";
    outfile << std::setw(21) << "acceptance_rate\n";

    for (int i = 0; i < n_variations_final; i++)
    {   /*
        Write data to file.
        */
        outfile << std::setw(20) << std::setprecision(10);
        outfile << alphas(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_variances(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_expectations(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << timing(i);
        outfile << std::setw(20) << std::setprecision(10);
        outfile << acceptances(i)/(n_particles*n_mc_cycles) << "\n";
    }
    outfile.close();
    std::cout << fpath << " written to file." << std::endl;
}

void VMC::write_energies_to_file(std::string fpath)
{   /*
    Write energy data to file.

    Parameters
    ----------
    fpath : std::string
        Relative file path and name.
    */
    outfile.open(fpath, std::ios::out);

    outfile << "alphas" << "\n";
    for (int i = 0; i < n_variations_final; i++){
      outfile << std::setw(20) << std::setprecision(10);
      outfile << alphas(i);
    }
    outfile << "\n";
    arma::Mat<double> energies_subview = energies.cols(0, n_variations_final - 1); 
    energies_subview.save(outfile, arma::raw_ascii);
    outfile.close();
    std::cout << fpath << " written to file." << std::endl;
}

void VMC::write_to_file_onebody_density(std::string fpath)
{   /*
    Write one-body density data to file.  Alphas are written as the
    first row. All following rows are particle occupation per bin data
    for the given alphas.

    Parameters
    ----------
    fpath : std::string
        Relative file path and name.
    */
    outfile.open(fpath, std::ios::out);

    for (int variation = 0; variation < n_variations; variation++)
    {
        outfile << std::setw(25) << alphas(variation);
    }
    outfile << "\n";
    particle_per_bin_count.save(outfile, arma::raw_ascii);
    outfile.close();
    std::cout << fpath << " written to file." << std::endl;
}

VMC::~VMC()
{
}
