#ifndef OTHER
#define OTHER

#include <armadillo>

const double hbar = 1;
const double m = 1;
const double omega = 1;

double spherical_harmonic_oscillator(double x, double y, double z, double omega);
inline double local_energy_3d(double x, double y, double z, double alpha, double beta);   //
double local_energy_3d(arma::Mat<double> pos, double alpha, double beta);          //
double local_energy_2d(arma::Mat<double> pos, double alpha, double beta);          //
double local_energy_1d(arma::Mat<double> pos, double alpha, double beta);          //
double quantum_force(double x, double y, double z, double alpha, double beta);

#endif
