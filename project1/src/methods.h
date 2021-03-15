#ifndef METHODS
#define METHODS
#include "VMC.h"

class BruteForce : public VMC
{
    // using VMC::VMC; // Inherit constructor of VMC class.
    private:
        const double step_size;
    public:
        BruteForce(
            const int n_dims_input,
            const int n_variations_input,
            const int n_mc_cycles_input,
            const int n_particles_input,
            arma::Col<double> alphas,
            const double brute_force_step_size_input
        );
        void one_variation(int variation);
};

class ImportanceSampling : public VMC
{

    protected:
        double wave_derivative = 0;
        double wave_derivative_expectation = 0;
        double wave_times_energy_expectation = 0;
        const double time_step;
    public:
        ImportanceSampling(
            const int n_dims_input,
            const int n_variations_input,
            const int n_mc_cycles_input,
            const int n_particles_input,
            arma::Col<double> alphas,
            const double importance_time_step_input
        );
        void one_variation(int variation);
};

class GradientDescent : public ImportanceSampling
{
    using ImportanceSampling::ImportanceSampling;
    private:
        // double energy_derivative = 0;
    public:
        void solve();
        // void metropolis(int dim, int particle, double alpha, int &acceptance);
};

#endif