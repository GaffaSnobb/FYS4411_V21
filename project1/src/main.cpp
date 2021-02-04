#include <iostream>         // Printing.
#include <cmath>            // Math library.
#include <random>           // RNG.
#include <fstream>          // Write to file.
#include <iomanip>          // Data formatting when writing to file.
#include <chrono>           // Timing.
#include <armadillo>        // Linear algebra.
#include "omp.h"            // Parallelization.
#include "wave_function.cpp"
#include "other_functions.cpp"

// These are currently defined in 'other_functions.cpp'.
// const double hbar = 1;
// const double m = 1;
// const double omega = 1;
// const double beta = 1;

class VMC
{
private:
    std::string fpath;              // Path to output text file.
    std::ofstream outfile;          // Output file.
    const int n_variations = 100;   // Number of variations.
    const int n_mc_cycles = 1e4;    // Number of MC cycles.
    const int seed = 1337;          // RNG seed.
    const int n_particles = 100;    // Number of particles.
    const int n_dims = 3;           // Number of spatial dimensions.
    const double step_size = 1;
    const double alpha_step = 0.1;
    
    double e_expectation_squared;   // Square of the energy expectation value.
    double de;                      // Energy step size.
    double exponential_diff;        // Difference of the exponentials, for Metropolis.  

    arma::Mat<double> pos_new;          // Proposed new position.
    arma::Mat<double> pos_current;      // Current position.
    arma::Col<double> wave_current;     // Current wave function.
    arma::Col<double> wave_new;         // Proposed new wave function.
    arma::Col<double> e_variances;      // Energy variances.
    arma::Col<double> e_expectations;   // Energy expectation values.
    arma::Col<double> alphas;           // Variational parameter.

    std::mt19937 engine;      // Mersenne Twister RNG.
    std::uniform_real_distribution<double> uniform;  // Continuous uniform distribution.

public:
    VMC()
    {
        fpath = "generated_data/output.txt";                    // Path to output text file.
        pos_new = arma::Mat<double>(n_dims, n_particles);       // Proposed new position.
        pos_current = arma::Mat<double>(n_dims, n_particles);   // Current position.
        wave_current = arma::Col<double>(n_particles);          // Current wave function.
        wave_new = arma::Col<double>(n_particles);              // Proposed new wave function.
        e_variances = arma::Col<double>(n_variations);          // Energy variances.
        e_expectations = arma::Col<double>(n_variations);       // Energy expectation values.
        e_expectations.zeros();
        
        // Pre-filling the alphas vector due to parallelization.
        alphas = arma::Col<double>(n_variations);   // Variational parameter.
        alphas.fill(alpha_step);
        alphas = arma::cumsum(alphas);

        engine.seed(seed);
    }

    void brute_force()
    {
        //Declared outside loop due to parallelization.
        int particle;   // Index for particle loop.
        int _;          // Index for MC loop.

        for (int i = 0; i < n_variations; i++)
        {   /*
            Run over all variations.
            */
            e_expectation_squared = 0;

            for (particle = 0; particle < n_particles; particle++)
            {   /*
                Iterate over all particles.  The dim iteration is hard-
                coded to avoid loop overhead.
                */
                pos_current(0, particle) = step_size*(uniform(engine) - 0.5);
                pos_current(1, particle) = step_size*(uniform(engine) - 0.5);
                pos_current(2, particle) = step_size*(uniform(engine) - 0.5);
                wave_current(particle) =
                    wave_function_exponent(
                        pos_current(0, particle),   // x.
                        pos_current(1, particle),   // y.
                        pos_current(2, particle),   // z.
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
                    Iterate over all particles.  The dim iteration is hard-
                    coded to avoid loop overhead.
                    */
                    pos_new(0, particle) = pos_current(0, particle) + step_size*(uniform(engine) - 0.5);
                    pos_new(1, particle) = pos_current(1, particle) + step_size*(uniform(engine) - 0.5);
                    pos_new(2, particle) = pos_current(2, particle) + step_size*(uniform(engine) - 0.5);
                    wave_new[particle] =
                        wave_function_exponent(
                            pos_new(0, particle),   // x.
                            pos_new(1, particle),   // y.
                            pos_new(2, particle),   // z.
                            alphas(i),
                            beta
                        );

                    exponential_diff = 2*(wave_new(particle) - wave_current(particle));

                    if (uniform(engine) < std::exp(exponential_diff))
                    {   /*
                        Perform the Metropolis algorithm.  To save one exponential
                        calculation, the difference is taken of the exponents
                        instead of the ratio of the exponentials. Marginally
                        better...
                        */
                        pos_current(0, particle) = pos_new(0, particle);
                        pos_current(1, particle) = pos_new(1, particle);
                        pos_current(2, particle) = pos_new(2, particle);
                        wave_current(particle) = wave_new(particle);
                    }

                    de = local_energy_3d(
                        pos_current(0, particle),
                        pos_current(1, particle),
                        pos_current(2, particle),
                        alphas(i)
                    );
                    // de = local_energy_3d(
                    //     pos_current(0, particle),
                    //     0,
                    //     0,
                    //     alphas(i)
                    // );
                    // de = local_energy_1d(pos_current(0, particle), alphas(i));

                    e_expectations(i) += de;
                    e_expectation_squared += de*de;
                }
            }

            e_expectations(i) /= n_mc_cycles;
            e_expectation_squared /= n_mc_cycles;
            e_variances(i) = e_expectation_squared - e_expectations(i)*e_expectations(i);
        }
    }

    void write_to_file()
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

    ~VMC()
    {

    }
};


int main()
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    
    VMC q;
    q.brute_force();
    q.write_to_file();

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;


    return 0;
}