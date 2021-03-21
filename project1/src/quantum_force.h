#ifndef QUANTUM
#define QUANTUM

#include <armadillo>
arma::Mat<double> quantum_force_3d_no_interaction(arma::Mat<double> pos, double alpha);
arma::Mat<double> quantum_force_3d_interaction(arma::Mat<double> pos, double alpha);

#endif