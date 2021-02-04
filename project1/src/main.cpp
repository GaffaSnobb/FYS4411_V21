#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <armadillo>

const int hbar = 1;
const double m = 1;
const double omega = 1;


double spherical_harmonic_oscillator(double x, double y, double z, double omega)
{
    return 0.5*m*omega*(x*x + y*y + z*z);
}

double wave_function(double x, double y, double z, double alpha, double beta)
{
    // double result = 1;
    // for (int _ = 0; _ < n; _++)
    // {
    //     result *= std::exp(-alpha*(x*x + y*y + beta*z*z));
    // }

    // Interaction term:
    // Double loop for f(...)

    // return result;
    return std::exp(-alpha*(x*x + y*y + beta*z*z));
}

double wave_function_exponent(double x, double y, double z, double alpha, double beta)
{
    return -alpha*(x*x + y*y + beta*z*z);
}

double local_energy_1d(double x, double alpha)
{   /*
    Analytical expression for the local energy for n particles, 1 dimension.
    */
    return -hbar*hbar*alpha/m*(2*alpha*x*x - 1) + 0.5*m*omega*x*x;
}

double local_energy_3d(double x, double y, double z, double alpha)
{
    return -hbar*hbar*alpha/m*(2*alpha*x*x - 1) + 0.5*m*omega*x*x;
}

void monte_carlo()
{
    char fpath[] = "generated_data/output.txt";
    const int n_variations = 100;
    const int n_mc_cycles = 1e4;
    const int seed = 1337;
    const int n_particles = 100;        // Number of particles.
    const int n_dims = 3;               // Number of spatial dimenstions.
    const double beta = 0;

    double step_size = 1;                      // Step size.
    double alpha = 0;                   // Intial value.
    double e_expectation_squared;
    double de;                          // Energy step size.
    double exponential_diff;

    arma::Mat<double> pos_new(n_dims, n_particles);
    arma::Mat<double> pos_current(n_dims, n_particles);

    arma::Col<double> alphas(n_variations);
    arma::Col<double> wave_current(n_particles);    // Current wave function.
    arma::Col<double> wave_new(n_particles);        // Proposed new wave function.
    arma::Col<double> e_variances(n_variations);    // Energy variances.
    arma::Col<double> e_expectations(n_variations); // Energy expectation values.
    e_expectations.zeros();

    std::ofstream outfile;
    std::mt19937 engine(seed);      // Seed the random engine which uses mersenne twister.
    std::uniform_real_distribution<double> uniform;  // Create continuous uniform distribution.

    for (int i = 0; i < n_variations; i++)
    {   /*
        Run over all variations.
        */
        alpha += 0.02;
        alphas(i) = alpha;

        e_expectation_squared = 0;

        for (int particle = 0; particle < n_particles; particle++)
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

        for (int _ = 0; _ < n_mc_cycles; _++)
        {   /*
            Run over all Monte Carlo cycles.
            */
            for (int particle = 0; particle < n_particles; particle++)
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
                // de = local_energy_1d(pos_current[0][particle], alphas(i));

                e_expectations(i) += de;
                e_expectation_squared += de*de;
            }
        }

        e_expectations(i) /= n_mc_cycles;
        e_expectation_squared /= n_mc_cycles;
        e_variances(i) = e_expectation_squared - e_expectations(i)*e_expectations(i);

    }

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


int main()
{
    // arma::mat A(5, 5);
    // A.print();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    monte_carlo();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;


    return 0;
}