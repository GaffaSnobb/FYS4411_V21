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
            const double beta_input,
            const double brute_force_step_size_input,
            const bool numerical_differentiation_input,
            bool debug
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
            const double beta_input,
            const double importance_time_step_input,
            const bool numerical_differentiation_input,
            bool debug_input
        );
        void one_variation(int variation);
};

class GradientDescent : public ImportanceSampling
{
    private:
        const double learning_rate;
        const double initial_alpha;
    public:
        GradientDescent(
            const int n_dims_input,
            const int n_variations_input,
            const int n_mc_cycles_input,
            const int n_particles_input,
            const double importance_time_step_input,
            const double learning_rate_input,
            const double initial_alpha_input,
            const double beta_input,
            const bool numerical_differentiation_input,
            bool debug_input
        );
        void solve(const double tol);
};

#endif