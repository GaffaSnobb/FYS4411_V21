#ifndef WAVE
#define WAVE

double wave_function(double x, double y, double z, double alpha, double beta)
{
    return std::exp(-alpha*(x*x + y*y + beta*z*z));
}

double wave_function_exponent_1d(arma::Mat<double> pos, double alpha, double beta)
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
    return -alpha*pos(0)*pos(0);
}

double wave_function_exponent_2d(arma::Mat<double> pos, double alpha, double beta)
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

double wave_function_exponent_3d(arma::Mat<double> pos, double alpha, double beta)
{   /*
    Parameters
    ----------
    pos : arma::Col<double>
        x, y, z position of a particle.

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

double wave_function_3d_diff_wrt_alpha(arma::Mat<double> pos, double alpha, double beta)
{   /*
    The wavefunction in 3D differentiated with respect to the
    variational parameter alpha.

    Parameters
    ----------
    pos : arma::Col<double>
        x, y, z position of a particle.

    alpha : double
        Variational parameter.

    beta : double
        ??? parameter.

    Returns
    -------
    : double
        The wave function differentiated with respect to alpha evaluated
        at pos.
    */
    double r_beta = pos(0)*pos(0) + pos(1)*pos(1) + beta*pos(2)*pos(2);
    return -(r_beta)*std::exp(-alpha*r_beta);
}
#endif