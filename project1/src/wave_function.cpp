#include "wave_function.h"

double wave_function_1d_no_interaction_with_loop(
    arma::Mat<double> pos,
    double alpha,
    double beta,
    const int n_particles
)
{   /*
    Parameters
    ----------
    pos : arma::Col<double>
        Vector of position of all particles. 3xN.

    alpha : double
        Variational parameter.

    beta : double
        ??? parameter.

    n_particles : constant integer
        The total number of particles.

    Returns
    -------
    : double
        The resulting wave function.
    */

    double res = 0;

    for (int particle = 0; particle < n_particles; particle++)
    {
        res += -alpha*pos(0, particle)*pos(0, particle);
    }
    return std::exp(res);
}

autodiff::var wave_function_1d_no_interaction(
    autodiff::var x,
    double alpha,
    double beta
)
{   /*
    For numerical differentiation with autodiff.

    Parameters
    ----------
    pos : autodiff::var
        Position of a single particle.

    alpha : double
        Variational parameter.

    beta : double
        ??? parameter.

    Returns
    -------
    : autodiff::var
        Wave function evaluated for a single particle.
    */
    return autodiff::reverse::exp(-alpha*x*x);
}

double wave_function_exponent_2d_no_interaction(
    arma::Mat<double> pos,
    double alpha,
    double beta
)
{   /*
    Parameters
    ----------
    pos : arma::Col<double>
        Vector of position of a particle.

    alpha : double
        Variational parameter.

    beta : double
        ??? parameter.

    Returns
    -------
    : double
        The resulting wave function exponent.
    */
    return -alpha*(pos(0)*pos(0) + pos(1)*pos(1));
}

double wave_function_exponent_3d_no_interaction(
    arma::Mat<double> pos,
    double alpha,
    double beta
)
{   /*
    Parameters
    ----------
    pos : arma::Col<double>
        Vector of position of a particle.

    alpha : double
        Variational parameter.

    beta : double
        ??? parameter.

    Returns
    -------
    : double
        The resulting wave function exponent.
    */
    return -alpha*(pos(0)*pos(0) + pos(1)*pos(1) + beta*pos(2)*pos(2));
}

double wave_function_3d_no_interaction_with_loop(
    arma::Mat<double> pos,
    double alpha,
    double beta,
    const int n_particles
    )
{   /*
    Parameters
    ----------
    pos : arma::Col<double>
        Vector of position of a particle.

    alpha : double
        Variational parameter.

    beta : double
        ??? parameter.

    Returns
    -------
    : double
        The resulting wave function evaluated at pos.
    */

    double wave_function = 1;
    for (int particle = 0; particle < n_particles; particle++)
    {
        wave_function *= std::exp(-alpha*(
            pos(0, particle)*pos(0, particle) +
            pos(1, particle)*pos(1, particle) +
            pos(2, particle)*pos(2, particle)*beta
        ));
    }

    return wave_function;
}

double wave_function_3d_interaction_with_loop(
    arma::Mat<double> pos,
    double alpha,
    double beta,
    const int n_particles
)
{   /*
    3D wave function with interaction term.

    Parameters
    ----------
    pos : arma::Mat<double>
        Position matrix of all particles. 3xN.

    alpha : double
        Variational parameter.

    beta : double
        ???

    n_particles : constant integer
        The total number of particles.

    Returns
    -------
    wave_function*wave_function_inner : double
        The total wavefunction.
    */
    
    double a = 0.0043;  // Prob. not right, so fix this.
    double wave_function = 1;       // Non-interaction term.
    double wave_function_inner = 1; // Interaction term.
    double particle_distance;       // Condition for the interaction term of the wavefunction.

    int particle;       // Index for particle loop.
    int particle_inner; // Index for inner particle loop.

    for (particle = 0; particle < n_particles; particle++)
    {   /*
        No interaction term.
        */
        wave_function *= std::exp(-alpha*(
            pos(0, particle)*pos(0, particle) +
            pos(1, particle)*pos(1, particle) +
            pos(2, particle)*pos(2, particle)*beta
        ));
    }
    for (particle = 0; particle < n_particles; particle++)
    {   /*
        Interaction term.
        */

        for (particle_inner = particle + 1; particle_inner < n_particles; particle_inner++)
        {
            particle_distance =
                arma::norm(pos.col(particle) - pos.col(particle_inner), 2);

            if (particle_distance > a)
            {   /*
                Interaction if the particle spacing is greater than 'a'.
                If the spacing is 'a' or less, the interaction is 1, but
                we do not need to state that explicitly (there are
                invisible ones everywhere!!).
                */
                wave_function_inner *= 1 - a/particle_distance;
            }
        }
    }
    return wave_function*wave_function_inner;
}

double wave_function_3d_diff_wrt_alpha(
    arma::Mat<double> pos,
    double alpha,
    double beta
)
{   /*
    CORRECTION: This is only the factor in front of the wave function
    after differentiation. 

    Parameters
    ----------
    pos : arma::Col<double>
        Vector of position of a particle.

    alpha : double
        Variational parameter.

    beta : double
        ??? parameter.

    Returns
    -------
    : double
        The wave function differentiated with respect to alpha evaluated
        at pos divided by the wavefunction.
    */
    return -(pos(0)*pos(0) + pos(1)*pos(1) + beta*pos(2)*pos(2));
}
