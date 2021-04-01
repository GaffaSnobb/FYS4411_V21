#include "quantum_force.h"
#include "parameters.h"

arma::Mat<double> quantum_force_3d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Parameters
    ----------
    pos : arma::Mat<double> reference
        Positions of all particles.
    */
    return -4*alpha*pos.col(current_particle);
}
arma::Mat<double> quantum_force_3d_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
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

    int particle;       // Particle index.
    int particle_inner; // Particle index.

    arma::Col<double> term_1(3);

    const double x = pos(0, current_particle);  // Readability.
    const double y = pos(1, current_particle);
    const double z = pos(2, current_particle);

    // Term 1.
    term_1 = -4*alpha*pos.col(current_particle);
    // Term 1 end.

    // Term 2.
    arma::Col<double> term_2(3);
    term_2.zeros();
    double particle_distance_1;   // |r_k - r_j|.
    arma::Col<double> particle_diff_vector_1(3);  // r_k - r_j.

    for (particle = 0; particle < current_particle; particle++)
    {
        particle_distance_1 =
            arma::norm((pos.col(current_particle) - pos.col(particle)));

        particle_diff_vector_1 =
            (pos.col(current_particle) - pos.col(particle));   // /particle_distance_1;

        if (particle_distance_1 > a)
        {
            particle_diff_vector_1 *= a / (particle_distance_1*particle_distance_1*(particle_distance_1 - a));
        }
        else
        {
            particle_diff_vector_1 = {0, 0, 0};
        }

        term_2 += particle_diff_vector_1;
    }

    for (particle = current_particle + 1; particle < n_particles; particle++)
    {
        particle_distance_1 =
            arma::norm((pos.col(current_particle) - pos.col(particle)));

        particle_diff_vector_1 =
            (pos.col(current_particle) - pos.col(particle));  //  /particle_distance_1;

        if (particle_distance_1 > a)
        {
            particle_diff_vector_1 *= a / (particle_distance_1*particle_distance_1*(particle_distance_1 - a));
        }
        else
        {
            particle_diff_vector_1 = {0, 0, 0};
        }

        term_2 += particle_diff_vector_1;
    }

    term_2 *= 2;

    // Term 2 end.

    return term_1 + term_2;
}
