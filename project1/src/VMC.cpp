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
    pos_new = arma::Mat<double>(n_dims, n_particles);         // Proposed new position.
    pos_current = arma::Mat<double>(n_dims, n_particles);     // Current position.
    e_variances = arma::Col<double>(n_variations);            // Energy variances.
    e_expectations = arma::Col<double>(n_variations);         // Energy expectation values.
    alphas = arma::Col<double>(n_variations);                 // Variational parameter.
    qforce_current = arma::Mat<double>(n_dims, n_particles);  // Current quantum force.
    qforce_new = arma::Mat<double>(n_dims, n_particles);      // New quantum force.
    test_local = arma::Row<double>(n_mc_cycles);              // Temporary
    energies = arma::Mat<double>(n_mc_cycles, n_variations);

    acceptances = arma::Col<double>(n_variations);   // Debug.
    acceptances.zeros();

    // Pre-filling the alphas vector due to parallelization.
    alphas.fill(alpha_step);
    alphas = arma::cumsum(alphas);
    e_expectations.zeros(); // Array must be zeroed since values will be added.
    energies.zeros();
    engine.seed(seed);

    set_local_energy();
    set_wave_function();
}

void VMC::set_local_energy()
{   /*
    Set pointers to the correct local energy function.
    */
    //std::cout << "VMC.cpp: set_local_energy()" << std::endl;
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
    //std::cout << "VMC.cpp: set_wave_function()" << std::endl;
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

    for (int i = 0; i < n_variations; i++)
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