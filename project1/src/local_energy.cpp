#include "local_energy.h"
#include "parameters.h"
#include "omp.h"

double local_energy_3d_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Analytical expression for the local energy for 3 dimensions, with
    interaction between particles.

    Parameters
    ----------
    pos : arma::Mat<double> reference
        Reference to position matrix of all particles. 3xN.

    alpha : double
        Variational parameter.

    beta : double
        ???

    current_particle : constant integer
        The index of the current particle.

    n_particles : constant integer
        The total number of particles.
    */

    int particle;   // Particle loop index.
    int particle_inner;   // Particle loop index.

    double x = pos(0, current_particle);
    double y = pos(1, current_particle);
    double z = pos(2, current_particle);
    
    // Term 1.
    double term_1 = -2*alpha;
    term_1 *= (2 - 2*alpha*(x*x + y*y + beta*beta*z*z) + beta*beta);
    // Term 1 end.

    // Term 2.
    double term_2 = -2*2*alpha;
    double particle_distance_1;   // |r_k - r_j|.
    arma::Col<double> term_2_vector(3);
    term_2_vector.zeros();
    arma::Col<double> particle_diff_vector_1(3);  // r_k - r_j.

    for (particle = 0; particle < current_particle; particle++)
    {
        particle_distance_1 =
            arma::norm((pos.col(current_particle) - pos.col(particle)), 2);

        particle_diff_vector_1 =
            (pos.col(current_particle) - pos.col(particle))/particle_distance_1;

        if (particle_distance_1 > a)
        {   /*
            Interaction if the particle spacing is greater than 'a'.
            */
            particle_diff_vector_1 *=
                a/(1 - a/particle_distance_1)*1/(particle_distance_1*particle_distance_1);
        }
        else
        {
            particle_diff_vector_1 = {0, 0, 0};
        }

        term_2_vector += particle_diff_vector_1;
    }

    for (particle = current_particle + 1; particle < n_particles; particle++)
    {
        particle_distance_1 =
            arma::norm((pos.col(current_particle) - pos.col(particle)), 2);

        particle_diff_vector_1 =
            (pos.col(current_particle) - pos.col(particle))/particle_distance_1;

        if (particle_distance_1 > a)
        {   /*
            Interaction if the particle spacing is greater than 'a'.
            */
            particle_diff_vector_1 *=
                a/(1 - a/particle_distance_1)*1/(particle_distance_1*particle_distance_1);
        }
        else
        {
            particle_diff_vector_1 = {0, 0, 0};
        }

        term_2_vector += particle_diff_vector_1;
    }

    particle_diff_vector_1 = {x, y, beta*z};  // Reuse the vector, dont need to allocate a new one.
    term_2 *= arma::dot(term_2_vector, particle_diff_vector_1);
    // Term 2 end.

    // Term 3.
    double term_3 = 0;
    double particle_distance_2;
    double u_diff_1;
    double u_diff_2;
    arma::Col<double> particle_diff_vector_2(3);
    
    for (particle = 0; particle < current_particle; particle++)
    {
        particle_diff_vector_1 =
            pos.col(current_particle) - pos.col(particle);
        particle_distance_1 =
            arma::norm((pos.col(current_particle) - pos.col(particle)), 2);


        if (particle_distance_1 > a)
        {   /*
            Interaction if the particle spacing is greater than 'a'.
            */
            u_diff_1 =
                a/(1 - a/particle_distance_1)*1/(particle_distance_1*particle_distance_1);
        }
        else
        {
            u_diff_1 = 0;
        }
        
        for (particle_inner = 0; particle_inner < current_particle; particle_inner++)
        {
            particle_diff_vector_2 =
                pos.col(current_particle) - pos.col(particle_inner);
            particle_distance_2 =
                arma::norm((pos.col(current_particle) - pos.col(particle_inner)), 2);

            if (particle_distance_2 > a)
            {   /*
                Interaction if the particle spacing is greater than 'a'.
                */
                u_diff_2 =
                    a/(1 - a/particle_distance_2)*1/(particle_distance_2*particle_distance_2);
            }
            else
            {
                u_diff_2 = 0;
            }

            term_3 += arma::dot(
                particle_diff_vector_1,
                particle_diff_vector_2
            )/(particle_distance_1*particle_distance_2)*u_diff_1*u_diff_2;
        }

        for (particle_inner = current_particle + 1; particle_inner < n_particles; particle_inner++)
        {
            particle_diff_vector_2 =
                pos.col(current_particle) - pos.col(particle_inner);
            particle_distance_2 =
                arma::norm((pos.col(current_particle) - pos.col(particle_inner)), 2);

            if (particle_distance_2 > a)
            {   /*
                Interaction if the particle spacing is greater than 'a'.
                */
                u_diff_2 =
                    a/(1 - a/particle_distance_2)*1/(particle_distance_2*particle_distance_2);
            }
            else
            {
                u_diff_2 = 0;
            }

            term_3 += arma::dot(
                particle_diff_vector_1,
                particle_diff_vector_2
            )/(particle_distance_1*particle_distance_2)*u_diff_1*u_diff_2;
        }
    }

    for (particle = current_particle + 1; particle < n_particles; particle++)
    {
        particle_diff_vector_1 =
            pos.col(current_particle) - pos.col(particle);
        particle_distance_1 =
            arma::norm((pos.col(current_particle) - pos.col(particle)), 2);


        if (particle_distance_1 > a)
        {   /*
            Interaction if the particle spacing is greater than 'a'.
            */
            u_diff_1 =
                a/(1 - a/particle_distance_1)*1/(particle_distance_1*particle_distance_1);
        }
        else
        {
            u_diff_1 = 0;
        }
        
        for (particle_inner = 0; particle_inner < current_particle; particle_inner++)
        {
            particle_diff_vector_2 =
                pos.col(current_particle) - pos.col(particle_inner);
            particle_distance_2 =
                arma::norm((pos.col(current_particle) - pos.col(particle_inner)), 2);

            if (particle_distance_2 > a)
            {   /*
                Interaction if the particle spacing is greater than 'a'.
                */
                u_diff_2 =
                    a/(1 - a/particle_distance_2)*1/(particle_distance_2*particle_distance_2);
            }
            else
            {
                u_diff_2 = 0;
            }

            term_3 += arma::dot(
                particle_diff_vector_1,
                particle_diff_vector_2
            )/(particle_distance_1*particle_distance_2)*u_diff_1*u_diff_2;
        }

        for (particle_inner = current_particle + 1; particle_inner < n_particles; particle_inner++)
        {
            particle_diff_vector_2 =
                pos.col(current_particle) - pos.col(particle_inner);
            particle_distance_2 =
                arma::norm((pos.col(current_particle) - pos.col(particle_inner)), 2);

            if (particle_distance_2 > a)
            {   /*
                Interaction if the particle spacing is greater than 'a'.
                */
                u_diff_2 =
                    a/(1 - a/particle_distance_2)*1/(particle_distance_2*particle_distance_2);
            }
            else
            {
                u_diff_2 = 0;
            }

            term_3 += arma::dot(
                particle_diff_vector_1,
                particle_diff_vector_2
            )/(particle_distance_1*particle_distance_2)*u_diff_1*u_diff_2;
        }
    }
    // Term 3 end.

    // Term 4.
    double term_4 = 0;
    for (particle = 0; particle < current_particle; particle++)
    {   
        particle_distance_1 =
            arma::norm((pos.col(current_particle) - pos.col(particle)), 2);

        if (particle_distance_1 > a)
        {   /*
            Interaction if the particle spacing is greater than 'a'.
            */
            u_diff_1 =
                a/(1 - a/particle_distance_1)*1/(particle_distance_1*particle_distance_1);
            u_diff_1 *= 2/particle_distance_1;

            u_diff_2 = (a*a - 2*a*particle_distance_1)/(particle_distance_1*particle_distance_1*(particle_distance_1 - a)*(particle_distance_1 - a));
        }
        else
        {
            u_diff_1 = 0;
            u_diff_2 = 0;
        }

        term_4 += u_diff_1 + u_diff_2;
    }
    // Term 4 end.

    // double res = -hbar*hbar/(2*m)*(term_1 + term_2 + term_3 + term_4);
    double res = 0.5*(-(term_1 + term_2 + term_3 + term_4) + x*x + y*y + z*z*gamma_*gamma_);
    res += 0.5*m*(omega*omega*(x*x + y*y) + omega*omega*z*z);   // V_ext.

    return res;
}

double local_energy_3d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Calculate the local energy for a single particle. Analytical
    expression for the local energy for 3 dimensions, no interaction
    between particles.

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

    const double x = pos(0, current_particle);
    const double y = pos(1, current_particle);
    const double z = pos(2, current_particle);

    return -hbar*hbar*alpha/m*(2*alpha*(x*x + y*y + beta*beta*z*z) - 2 - beta) + 0.5*m*omega*omega*(x*x + y*y + z*z);
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
    double x = pos(0, current_particle);
    return -hbar*hbar*alpha/m*(2*alpha*x*x - 1) + 0.5*m*omega*omega*x*x;
}

double local_energy_1d_no_interaction_numerical_differentiation(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Numerical expression for the local energy for 1 dimensions, no
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

    Params params;
    params.alpha = alpha;
    params.beta = beta;

    autodiff::HigherOrderDual<2> x = pos(0, current_particle);
    autodiff::HigherOrderDual<2> u = wave_function_1d_no_interaction(x, params);

    autodiff::dual uxx = autodiff::forward::derivative(
        wave_function_1d_no_interaction,
        autodiff::forward::wrt<2>(x),
        autodiff::forward::at(x, params)
    );

    return -hbar*double(uxx)/(2*m*double(u.val)) + 0.5*m*omega*omega*pos(0, current_particle)*pos(0, current_particle);
}