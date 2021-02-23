#ifndef METHODS
#define METHODS
#include "VMC.h"

class BruteForce : public VMC
{
    using VMC::VMC; // Inherit constructor of VMC class.
    public:
        // method = "brute_force";
        void set_initial_positions(int dim, int particle, int variation);
        void set_new_positions(int dim, int particle, int variation);
        void metropolis(int dim, int particle);
};

class ImportanceSampling : public VMC
{
    using VMC::VMC;
    public:
        void set_initial_positions(int dim, int particle, int variation);
        void set_new_positions(int dim, int particle, int variation);
        void metropolis(int dim, int particle);
};

#endif