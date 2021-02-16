#ifndef OTHER
#define OTHER

const double hbar = 1;
const double m = 1;
const double omega = 1;

double spherical_harmonic_oscillator(double x, double y, double z, double omega)
{
    return 0.5*m*omega*(x*x + y*y + z*z);
}

inline double local_energy_3d(double x, double y, double z, double alpha, double beta)
{   /*
    Analytical expression for the local energy for 3 dimensions, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*(x*x + y*y + beta*beta*z*z) - 2 - beta) + 0.5*m*omega*omega*(x*x + y*y + z*z);
}

inline double local_energy_3d(arma::Mat<double> pos, double alpha, double beta)
{   /*
    Analytical expression for the local energy for 3 dimensions, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*(pos(0)*pos(0) + pos(1)*pos(1) + beta*beta*pos(2)*pos(2)) - 2 - beta) + 0.5*m*omega*omega*(pos(0)*pos(0) + pos(1)*pos(1) + pos(2)*pos(2));
}

inline double local_energy_2d(arma::Mat<double> pos, double alpha, double beta)
{   /*
    Analytical expression for the local energy for 2 dimensions, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*(pos(0)*pos(0) + pos(1)*pos(1)) - 2) + 0.5*m*omega*omega*(pos(0)*pos(0) + pos(1)*pos(1));
}

inline double local_energy_1d(arma::Mat<double> pos, double alpha, double beta)
{   /*
    Analytical expression for the local energy for 1 dimension, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*pos(0)*pos(0) - 1) + 0.5*m*omega*omega*pos(0)*pos(0);
}

double quantum_force(double x, double y, double z, double alpha, double beta)
{   /*
    Analytical expression for quantum force with beta=1 and a=0.
    MAY BE REMOVED
    */
    return -4.0 * alpha * (x + y + z);
}

double local_energy_3d_diff_wrt_alpha(arma::Mat<double> pos, double beta)
{   /*
    UNSURE IF NEEDED
    Local energy in 3D differentiated with respect to the variational
    parameter alpha.

    Parameters
    ----------
    pos : arma::Mat<double>
        x, y, z position of a particle.

    beta : double
        ??? parameter. (interaction parameter maybe?)

    Returns
    -------
    : double
        The local energy differentiated with respect to alpha evaluated
        at pos with parameter beta.

    */

    return -2*hbar*hbar/m*(pos(0)*pos(0) + pos(1)*pos(1) + beta*pos(2)*pos(2));

}

#endif
