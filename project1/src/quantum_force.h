#ifndef QUANTUM
#define QUANTUM

#include <armadillo>
void quantum_force_3d_no_interaction(arma::Mat<double> pos, arma::Mat<double> &qforce, double alpha, double beta, const int n_particles);
arma::Mat<double> quantum_force_3d_interaction(arma::Mat<double> pos, double alpha, double beta, const int n_particles);

#endif