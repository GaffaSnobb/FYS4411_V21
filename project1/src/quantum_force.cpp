#include "quantum_force.h"
#include <armadillo>

void quantum_force_3d_no_interaction(
    arma::Mat<double> pos,
    arma::Mat<double> &qforce,
    double alpha,
    double beta,
    const int n_particles
)
{   /*
    Parameters
    ----------
    pos : arma::Mat<double>
        Positions of all particles.
    */
    for (int particle = 0; particle < n_particles; particle++)
    {
        qforce.col(particle) = -4*alpha*pos.col(particle);
    }
}

// arma::Mat<double> quantum_force_3d_interaction(arma::Mat<double> pos, double alpha)
// {   /*
//     Parameters
//     ----------
//     pos : arma::Mat<double>
//         The position of a single particle
//     */
    
// }