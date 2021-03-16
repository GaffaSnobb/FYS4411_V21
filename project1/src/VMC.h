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
#include <string>   // String type, string maipulation.

class VMC
{
    protected:
        // TODO: Some of these values may not be needed by all classes. Move them appropriately.
        std::string fpath;              // Path to output text file. TODO: Set fpath according to what sub class is used. No input needed.
        std::ofstream outfile;          // Output file.
        const int n_variations;         // Number of variations.
        const int n_mc_cycles;          // Number of MC cycles.
        const int seed = 1337;          // RNG seed.
        const int n_particles;          // Number of particles.
        const int n_dims;               // Number of spatial dimensions.

        const double beta = 1;
        const double diffusion_coeff = 0.5;

        double energy_expectation_squared;  // Square of the energy expectation value.
        double local_energy;                // Local energy.
        double exponential_diff;            // Difference of the exponentials, for Metropolis.
        double wave_current;                // Current wave function.
        double wave_new;                    // Proposed new wave function.
        double energy_expectation = 0;
        double energy_variance = 0;


        int particle;       // Index for particle loop.
        int particle_inner; // Index for inner particle loops.
        int mc;             // Index for MC loop.
        int dim;            // Index for dimension loops.

        // Moved initialization to class constructor.
        arma::Mat<double> pos_new;       // Proposed new position.
        arma::Mat<double> pos_current;   // Current position.
        arma::Col<double> e_variances;   // Energy variances.
        arma::Col<double> e_expectations;// Energy expectation values.
        arma::Col<double> alphas;        // Variational parameter.
        arma::Mat<double> qforce_current;// Current quantum force.
        arma::Mat<double> qforce_new;    // New quantum force.

        arma::Row<double> test_local;    // Temporary
        arma::Mat<double> energies;

        std::mt19937 engine;      // Mersenne Twister RNG.
        std::uniform_real_distribution<double> uniform;  // Continuous uniform distribution.
        std::normal_distribution<double> normal;         // Gaussian distribution

        double (*local_energy_ptr)(arma::Mat<double>, double, double);  // Function pointer.
        double (*wave_function_exponent_ptr)(arma::Mat<double>, double, double);

    public:
        arma::Col<double> acceptances;   // Debug.
        VMC(
            const int n_dims_input,
            const int n_variations_input,
            const int n_mc_cycles_input,
            const int n_particles_input,
            arma::Col<double> alphas
        );
        void set_local_energy();
        void set_wave_function();
        void write_to_file(std::string fname);
        void write_to_file_particles(std::string fpath);
        void write_energies_to_file(std::string fpath);
        void solve();
        virtual void one_variation(int variation);
        ~VMC();
};

#endif
