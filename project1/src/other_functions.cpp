#ifndef OTHER
#define OTHER

const double hbar = 1;
const double m = 1;
const double omega = 1;
const double beta = 1;

double spherical_harmonic_oscillator(double x, double y, double z, double omega)
{
    return 0.5*m*omega*(x*x + y*y + z*z);
}


double local_energy_1d(double x, double alpha)
{   /*
    Analytical expression for the local energy for 1 dimension, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*x*x - 1) + 0.5*m*omega*x*x;
}

double local_energy_3d(double x, double y, double z, double alpha)
{   /*
    Analytical expression for the local energy for 3 dimensions, no
    interaction between particles.
    */
    return -hbar*hbar*alpha/m*(2*alpha*(x*x + y*y + beta*beta*z*z) - 2 - beta) + 0.5*m*omega*omega*(x*x + y*y + z*z);
}

#endif