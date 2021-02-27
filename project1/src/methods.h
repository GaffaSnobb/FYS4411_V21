#ifndef METHODS
#define METHODS
#include "VMC.h"

class BruteForce : public VMC
{
    using VMC::VMC; // Inherit constructor of VMC class.
    public:
        void solve();
        void one_variation(int variation);
};

class ImportanceSampling : public VMC
{
    using VMC::VMC;
    protected:
        double wave_derivative = 0;
        double wave_derivative_expectation = 0;
        double wave_times_energy_expectation = 0;
        double time_step = 0.01;
    public:
        void solve();
        void one_variation(int variation);
};

class GradientDescent : public ImportanceSampling
{
    using ImportanceSampling::ImportanceSampling;
    private:
        // double energy_derivative = 0;
    public:
        void solve();
        void metropolis(int dim, int particle, double alpha, int &acceptance);
        void extra(double alpha);
};

#endif