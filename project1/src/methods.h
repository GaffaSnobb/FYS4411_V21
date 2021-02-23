#ifndef METHODS
#define METHODS
#include "VMC.h"

class BruteForce : public VMC
{
    using VMC::VMC; // Inherit constructor of VMC class.
    public:
        // method = "brute_force";
        void set_initial_positions(int dim, int particle, double alpha);
        void set_new_positions(int dim, int particle, double alpha);
        void metropolis(int dim, int particle, double alpha);
        void solve();
};

class ImportanceSampling : public VMC
{
    using VMC::VMC;
    protected:
        double time_step = 0.1;
    public:
        void set_initial_positions(int dim, int particle, double alpha);
        void set_new_positions(int dim, int particle, double alpha);
        void metropolis(int dim, int particle, double alpha);
        void solve();
};

class GradientDescent : public ImportanceSampling
{
    using ImportanceSampling::ImportanceSampling;
    private:
        double wave_derivative = 0;
        double wave_derivative_expectation = 0;
        double wave_times_energy_expectation = 0;
        // double energy_derivative = 0;
    public:
        void solve();
        void metropolis(int dim, int particle, double alpha);
        void extra(double alpha);
};

#endif