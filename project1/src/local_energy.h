#ifndef OTHER
#define OTHER

#include <armadillo>

const double hbar = 1;
const double m = 1;
const double omega = 1;

double local_energy_3d(double x, double y, double z, double alpha, double beta);
double local_energy_3d_no_interaction(arma::Mat<double> pos, double alpha, double beta);
double local_energy_2d_no_interaction(arma::Mat<double> pos, double alpha, double beta);
double local_energy_1d_no_interaction(arma::Mat<double> pos, double alpha, double beta);

#endif