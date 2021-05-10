#include "wave_function.h"

double wave_function_1d_no_interaction_with_loop(
    const arma::Mat<double> &pos,
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

autodiff::HigherOrderDual<2> wave_function_1d_no_interaction(
    autodiff::HigherOrderDual<2> &x,
    const struct Params &params
)
{   /*
    For numerical differentiation with autodiff.

    Parameters
    ----------
    x : autodiff::HigherOrderDual<2>
        x position of a single particle.

    params : Params struct
        For passing function parameters to autodiff::forward enabled
        funstions. params contains:

        alpha : double
            Variational parameter.

        beta : double
            ??? parameter.

    Returns
    -------
    : autodiff::HigherOrderDual<2>
        Wave function evaluated for a single particle at a single point.
    */
    return autodiff::forward::exp(-params.alpha*x*x);
}

autodiff::HigherOrderDual<2> wave_function_2d_no_interaction(
    autodiff::HigherOrderDual<2> &x,
    autodiff::HigherOrderDual<2> &y,
    const struct Params &params
)
{   /*
    For numerical differentiation with autodiff.

    Parameters
    ----------
    x : autodiff::HigherOrderDual<2>
        x position of a single particle.

    y : autodiff::HigherOrderDual<2>
        y position of a single particle.

    params : Params struct
        For passing function parameters to autodiff::forward enabled
        funstions. params contains:

        alpha : double
            Variational parameter.

        beta : double
            ??? parameter.

    Returns
    -------
    : autodiff::HigherOrderDual<2>
        Wave function evaluated for a single particle at a single point.
    */
    return autodiff::forward::exp(-params.alpha*(x*x + y*y));
}

autodiff::HigherOrderDual<2> wave_function_3d_no_interaction(
    autodiff::HigherOrderDual<2> &x,
    autodiff::HigherOrderDual<2> &y,
    autodiff::HigherOrderDual<2> &z,
    const struct Params &params
)
{   /*
    For numerical differentiation with autodiff.

    Parameters
    ----------
    x : autodiff::HigherOrderDual<2>
        x position of a single particle.

    y : autodiff::HigherOrderDual<2>
        y position of a single particle.

    z : autodiff::HigherOrderDual<2>
        z position of a single particle.

    params : Params struct
        For passing function parameters to autodiff::forward enabled
        funstions. params contains:

        alpha : double
            Variational parameter.

        beta : double
            ??? parameter.

    Returns
    -------
    : autodiff::HigherOrderDual<2>
        Wave function evaluated for a single particle at a single point.
    */
    return autodiff::forward::exp(-params.alpha*(x*x + y*y + params.beta*z*z));
}

double wave_function_2d_no_interaction_with_loop(
    const arma::Mat<double> &pos,
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
    double x;
    double y;

    for (int particle = 0; particle < n_particles; particle++)
    {   
        x = pos(0, particle);
        y = pos(1, particle);
        res += -alpha*(x*x + y*y);
    }
    return std::exp(res);
}

double wave_function_exponent_3d_no_interaction(
    const arma::Mat<double> &pos,
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
    const arma::Mat<double> &pos,
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
    const arma::Mat<double> &pos,
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
    
    double wave_function = 0;   // Non-interaction term.
    double wave_function_inner = 1; // Interaction term.
    double particle_distance;       // Condition for the interaction term of the wavefunction.

    int particle;       // Index for particle loop.
    int particle_inner; // Index for inner particle loop.

    for (particle = 0; particle < n_particles; particle++)
    {   /*
        No interaction term.
        */
        wave_function += (-alpha*(
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
                arma::norm(pos.col(particle) - pos.col(particle_inner));

            if (particle_distance > a)
            {   /*
                Interaction if the particle spacing is greater than 'a'.
                If the spacing is 'a' or less, the interaction is 1, but
                we do not need to state that explicitly (there are
                invisible ones everywhere!!).
                */
                wave_function_inner *= 1 - a/particle_distance;
            }
            else
            {
                wave_function_inner = 0;
                break;  // Entire term is zero anyway.
            }
        }
    }
    return std::exp(wave_function)*wave_function_inner;
}

double wave_function_3d_diff_wrt_alpha(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta
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

double wave_function_2d_diff_wrt_alpha(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta
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
    return -(pos(0)*pos(0) + pos(1)*pos(1));
}

double wave_function_1d_diff_wrt_alpha(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta
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
    return -pos(0)*pos(0);
}