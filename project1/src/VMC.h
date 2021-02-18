#ifndef VMC_H
#define VMC_H

#include <iostream>         // Printing.
#include <cmath>            // Math library.
#include <random>           // RNG.
#include <fstream>          // Write to file.
#include <iomanip>          // Data formatting when writing to file.
#include <chrono>           // Timing.
#include <armadillo>        // Linear algebra.
#include "omp.h"            // Parallelization.
#include "wave_function.h"
#include "other_functions.h"

#include <sstream>
#include <string>

class VMC
{
private:
    std::string fpath;              // Path to output text file.
    std::ofstream outfile;          // Output file.
    const int n_variations = 100;   // Number of variations.
    const int n_mc_cycles = 70;     // Number of MC cycles.
    const int seed = 1337;          // RNG seed.
    const int n_particles = 100;    // Number of particles.
    const int n_dims;               // Number of spatial dimensions.
    const int method;

    const double step_size = 1;
    const double alpha_step = 0.02;
    const double beta = 1;
    const double diffusion_coeff = 0.5;

    // const double time_step = 0.4;
    double time_step;

    double e_expectation_squared;   // Square of the energy expectation value.
    double energy_step;             // Energy step size.
    double exponential_diff;        // Difference of the exponentials, for Metropolis.

    arma::Mat<double> pos_new = arma::Mat<double>(n_dims, n_particles);         // Proposed new position.
    arma::Mat<double> pos_current = arma::Mat<double>(n_dims, n_particles);     // Current position.
    arma::Col<double> wave_current = arma::Col<double>(n_particles);            // Current wave function.
    arma::Col<double> wave_new = arma::Col<double>(n_particles);                // Proposed new wave function.
    arma::Col<double> e_variances = arma::Col<double>(n_variations);            // Energy variances.
    arma::Col<double> e_expectations = arma::Col<double>(n_variations);         // Energy expectation values.
    arma::Col<double> alphas = arma::Col<double>(n_variations);                 // Variational parameter.
    arma::Mat<double> qforce_current = arma::Mat<double>(n_dims, n_particles);  // Current quantum force.
    arma::Mat<double> qforce_new = arma::Mat<double>(n_dims, n_particles);      // New quantum force.

    std::mt19937 engine;      // Mersenne Twister RNG.
    std::uniform_real_distribution<double> uniform;  // Continuous uniform distribution.
    std::normal_distribution<double> normal;         // Gaussian distribution

    double (*local_energy_ptr)(arma::Mat<double>, double, double);  // Function pointer.
    double (*wave_function_exponent_ptr)(arma::Mat<double>, double, double);

public:
    VMC(const int n_dims_input, const int method_input) : n_dims(n_dims_input),
                                                          method(method_input)
    {
        // Pre-filling the alphas vector due to parallelization.
        alphas.fill(alpha_step);
        alphas = arma::cumsum(alphas);

        e_expectations.zeros(); // Array must be zeroed since values will be added.
        engine.seed(seed);

        //n_dims = n_dims_input;
        std::cout << "VMC() in VMC.h  n_dims = " << n_dims << "  method = "<< method << std::endl;
    }

    void set_local_energy();
    void set_wave_function();
    void set_initial_positions(int dim, int particle);
    void set_new_positions(int dim, int particle);
    void brute_force();
    void importance_sampling(double t);
    void write_to_file(std::string fname);
};

#endif