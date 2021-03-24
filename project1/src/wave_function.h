#ifndef WAVE
#define WAVE

#include "VMC.h"
double wave_function_1d_no_interaction_with_loop(
    arma::Mat<double> pos,
    double alpha,
    double beta,
    const int n_particles
);
autodiff::var wave_function_1d_no_interaction_with_loop(
    autodiff::var x,
    double alpha,
    double beta
);
double wave_function_exponent_2d_no_interaction(
    arma::Mat<double> pos,
    double alpha,
    double beta
);
double wave_function_exponent_3d_no_interaction(
    arma::Mat<double> pos,
    double alpha,
    double beta
);
double wave_function_3d_no_interaction_with_loop(
    arma::Mat<double> pos,
    double alpha,
    double beta,
    const int n_particles
);
double wave_function_3d_interaction_with_loop(
    arma::Mat<double> pos,
    double alpha,
    double beta,
    const int n_particles
);
double wave_function_3d_diff_wrt_alpha(
    arma::Mat<double> pos,
    double alpha,
    double beta
);

#endif
