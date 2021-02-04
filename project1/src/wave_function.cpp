#ifndef WAVE
#define WAVE

double wave_function(double x, double y, double z, double alpha, double beta)
{
    return std::exp(-alpha*(x*x + y*y + beta*z*z));
}

double wave_function_exponent(double x, double y, double z, double alpha, double beta)
{
    return -alpha*(x*x + y*y + beta*z*z);
}

#endif