#ifndef WAVE
#define WAVE

#include "VMC.h"
#include "parameters.h"
double wave_function_1d_no_interaction_with_loop(
    const arma::Mat<double> &pos,
    double alpha,
    double beta,
    const int n_particles
);
autodiff::HigherOrderDual<2> wave_function_1d_no_interaction(
    autodiff::HigherOrderDual<2> &x,
    const struct Params &params
);
autodiff::HigherOrderDual<2> wave_function_2d_no_interaction(
    autodiff::HigherOrderDual<2> &x,
    autodiff::HigherOrderDual<2> &y,
    const struct Params &params
);
autodiff::HigherOrderDual<2> wave_function_3d_no_interaction(
    autodiff::HigherOrderDual<2> &x,
    autodiff::HigherOrderDual<2> &y,
    autodiff::HigherOrderDual<2> &z,
    const struct Params &params
);
double wave_function_2d_no_interaction_with_loop(
    const arma::Mat<double> &pos,
    double alpha,
    double beta,
    const int n_particles
);
double wave_function_exponent_3d_no_interaction(
    const arma::Mat<double> &pos,
    double alpha,
    double beta
);
double wave_function_3d_no_interaction_with_loop(
    const arma::Mat<double> &pos,
    double alpha,
    double beta,
    const int n_particles
);
double wave_function_3d_interaction_with_loop(
    const arma::Mat<double> &pos,
    double alpha,
    double beta,
    const int n_particles
);
double wave_function_3d_diff_wrt_alpha(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta
);
double wave_function_2d_diff_wrt_alpha(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta
);
double wave_function_1d_diff_wrt_alpha(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta
);
#endif
