#ifndef OTHER
#define OTHER

const double hbar = 1;
const double m = 1;
const double omega = 1;

double spherical_harmonic_oscillator(double x, double y, double z, double omega)
{
    return 0.5*m*omega*(x*x + y*y + z*z);
}

double local_energy_3d(double x, double y, double z, double alpha, double beta)
{   /*
    Analytical expression for the local energy for 3 dimensions, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*(x*x + y*y + beta*beta*z*z) - 2 - beta) + 0.5*m*omega*omega*(x*x + y*y + z*z);
}

double local_energy_2d(double x, double y, double alpha)
{   /*
    Analytical expression for the local energy for 2 dimensions, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*(x*x + y*y) - 2) + 0.5*m*omega*omega*(x*x + y*y);
}

double local_energy_1d(double x, double alpha)
{   /*
    Analytical expression for the local energy for 1 dimension, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*x*x - 1) + 0.5*m*omega*omega*x*x;
}

inline double local_energy(arma::Mat<double> pos, double alpha, double beta, const int n_dims)
{   /*
    Wrapper function for choosing the correct dimensionality.
    */
    if (n_dims == 1)
    {
        return local_energy_1d(
            pos(0),
            alpha
        );
    }
    else if (n_dims == 2)
    {
        return local_energy_2d(
            pos(0),
            pos(1),
            alpha
        );
    }
    else
    {
        return local_energy_3d(
            pos(0),
            pos(1),
            pos(2),
            alpha,
            beta
        );
    }
}

double quantum_force(double x, double y, double z, double alpha, double beta)
{   /*
    Analytical expression for quantum force with beta=1 and a=0.
    MAY BE REMOVED
    */
    return -4.0 * alpha * (x + y + z);
}

#endif
