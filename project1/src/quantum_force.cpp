#include "quantum_force.h"
#include <armadillo>

arma::Mat<double> quantum_force_3d_no_interaction(arma::Mat<double> pos, double alpha)
{   
    return -4*alpha*pos;
}

// arma::Mat<double> quantum_force_3d_interaction(arma::Mat<double> pos, double alpha)
// {   
//     return
// }