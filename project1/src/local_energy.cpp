#include "local_energy.h"

// double local_energy_3d_interaction(arma::Mat<double> pos, double alpha, double beta)
// {   /*
//     Analytical expression for the local energy for 3 dimensions, with
//     interaction between particles.
//     */

//     // Term 1.
//     double term_1 = -2*alpha*(2 - 2*alpha*(p))
//     // Term 1 end.
// }

double local_energy_3d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Analytical expression for the local energy for 3 dimensions, no
    interaction between particles.

    Parameters
    ----------
    pos : arma::Mat<double> reference
        Positions of all particles.

    alpha : constant double
        Current variational parameter.

    beta : constant double
        ???

    current_particle : constant integer
        The index of the current particle.

    n_particles : constant integer
        The total number of particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*(pos(0, current_particle)*pos(0, current_particle) + pos(1, current_particle)*pos(1, current_particle) + beta*beta*pos(2, current_particle)*pos(2, current_particle)) - 2 - beta) + 0.5*m*omega*omega*(pos(0, current_particle)*pos(0, current_particle) + pos(1, current_particle)*pos(1, current_particle) + pos(2, current_particle)*pos(2, current_particle));
}

double local_energy_2d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Analytical expression for the local energy for 2 dimensions, no
    interaction between particles.

    Parameters
    ----------
    pos : arma::Mat<double> reference
        Positions of all particles.

    alpha : constant double
        Current variational parameter.

    beta : constant double
        ???

    current_particle : constant integer
        The index of the current particle.

    n_particles : constant integer
        The total number of particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*(pos(0, current_particle)*pos(0, current_particle) + pos(1, current_particle)*pos(1, current_particle)) - 2) + 0.5*m*omega*omega*(pos(0, current_particle)*pos(0, current_particle) + pos(1, current_particle)*pos(1, current_particle));
}

double local_energy_1d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Analytical expression for the local energy for 1 dimensions, no
    interaction between particles.

    Parameters
    ----------
    pos : arma::Mat<double> reference
        Positions of all particles.

    alpha : constant double
        Current variational parameter.

    beta : constant double
        ???

    current_particle : constant integer
        The index of the current particle.

    n_particles : constant integer
        The total number of particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*pos(0, current_particle)*pos(0, current_particle) - 1) +
        0.5*m*omega*omega*pos(0, current_particle)*pos(0, current_particle);
}