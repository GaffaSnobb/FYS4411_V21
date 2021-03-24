#ifndef OTHER
#define OTHER

#include "VMC.h"

double local_energy_3d_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);
double local_energy_3d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);
double local_energy_2d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);
double local_energy_1d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);
double local_energy_1d_no_interaction_numerical_differentiation(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);
double local_energy_2d_no_interaction_numerical_differentiation(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);
double local_energy_3d_no_interaction_numerical_differentiation(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
);
#endif