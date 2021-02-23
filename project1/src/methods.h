#ifndef BRUTE_FORCE
#define BRUTE_FORCE
#include "VMC.h"

class BruteForce : public VMC
{
    using VMC::VMC; // Inherit constructor of VMC class.
    public:
        void set_initial_positions(int dim, int particle);
        void set_new_positions(int dim, int particle);
};

#endif