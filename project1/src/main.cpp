#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>

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

double local_energy(double x, double alpha)
{   /*
    Analytical expression for the local energy for n particles, 1 dimension.
    */
    // double result = 0;

    // for (int i = 0; i < n; i++)
    // {
    //     result += -hbar/m*alpha*(2*alpha*x[i]*x[i] - 1) + 0.5*m*omega*x[i]*x[i];
    // }

    // return result;
    return -hbar*hbar*alpha/m*(2*alpha*x*x - 1) + 0.5*m*omega*x*x;
}


void monte_carlo()
{
    const int n_variations = 100;
    const int n_mc_cycles = 1e5;
    const int seed = 1337;
    const double y = 0;
    const double z = 0;
    const double beta = 0;

    double dx = 1;                  // Step size.
    double x_current = 0;           // Current step.
    double x_new = 0;               // Proposed new step.
    double alphas[n_variations];
    double e_variances[n_variations];
    double e_expectations[n_variations] = {0};
    double alpha = 0;             // Intial value.
    double e_expectation_squared;
    double de;                      // Energy step size.
    double wave_current;            // Current wave function.
    double wave_new;                // Proposed new wave function.

    double exponential_diff;

    std::ofstream outfile;
    
    std::mt19937 engine(seed);      // Seed the random engine which uses mersenne twister.
    std::uniform_real_distribution<double> uniform;  // Create continuous uniform distribution.

    for (int i = 0; i < n_variations; i++)
    {   
        alpha += 0.05;
        alphas[i] = alpha;

        e_expectation_squared = 0;
        x_current = dx*(uniform(engine) - 0.5);
        wave_current = wave_function_exponent(x_current, y, z, alphas[i], beta);
        // wave_current = wave_function(x_current, y, z, alphas[i], beta);

        for (int _ = 0; _ < n_mc_cycles; _++)
        {   /*
            Run over all Monte Carlo cycles.
            */
            
            x_new = x_current + dx*(uniform(engine) - 0.5);
            // wave_new = wave_function(x_new, y, z, alphas[i], beta);
            wave_new = wave_function_exponent(x_new, y, z, alphas[i], beta);
            exponential_diff = 2*(wave_new - wave_current);

            // if (uniform(engine) < (wave_new*wave_new/(wave_current*wave_current)))
            if (uniform(engine) < std::exp(exponential_diff))
            {   /*
                Perform the Metropolis algorithm.  To save one exponential
                calculation, the difference is taken of the exponents instead
                of the ratio of the exponentials. 
                */
                x_current = x_new;
                wave_current = wave_new;
            }

            de = local_energy(x_current, alphas[i]);
            e_expectations[i] += de;
            e_expectation_squared += de*de;
        }

        e_expectations[i] /= n_mc_cycles;
        e_expectation_squared /= n_mc_cycles;
        e_variances[i] = e_expectation_squared - e_expectations[i]*e_expectations[i];

    }

    outfile.open("outfile.txt", std::ios::out);
    outfile << std::setw(20) << "alpha";
    outfile << std::setw(20) << "var";
    outfile << std::setw(21) << "exp\n";

    for (int i = 0; i < n_variations; i++)
    {   /*
        Write data to file.
        */
        outfile << std::setw(20) << std::setprecision(10);
        outfile << alphas[i];
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_variances[i];
        outfile << std::setw(20) << std::setprecision(10);
        outfile << e_expectations[i] << "\n";
    }


    outfile.close();
}


int main()
{

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    monte_carlo();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;

    
    return 0;
}