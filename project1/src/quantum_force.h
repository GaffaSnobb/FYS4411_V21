#ifndef QUANTUM
#define QUANTUM
#include "VMC.h"

arma::Mat<double> quantum_force_3d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);
arma::Mat<double> quantum_force_3d_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);

#endif