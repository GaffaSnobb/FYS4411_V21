#ifndef WAVE
#define WAVE

#include <armadillo>
double wave_function(double x, double y, double z, double alpha, double beta);
double wave_function_exponent_1d(arma::Mat<double> pos, double alpha, double beta);
double wave_function_exponent_2d(arma::Mat<double> pos, double alpha, double beta);
double wave_function_exponent_3d(arma::Mat<double> pos, double alpha, double beta);
double wave_function_3d_diff_wrt_alpha(arma::Mat<double> pos, double alpha, double beta);

#endif
