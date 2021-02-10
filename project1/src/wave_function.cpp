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

double wave_function_exponent(arma::Col<double> pos, double alpha, double beta, int n_dims)
{   /*
    Parameters
    ----------
    pos : arma::Col<double>
        Vector of x, y, z position of a particle.

    alpha : double
        Variational parameter.

    beta : double
        ??? parameter.

    n_dims : int
        The number of spatial dimensions and the length of pos vector.

    Returns
    -------
    result : double
        The resulting wave function exponent.
    */
    double result = 0;

    for (int dim = 0; dim < n_dims; dim++)
    {   
        if (dim == 2)
        {   /*
            z dimension has an additional beta factor.
            */
            result += pos(dim)*pos(dim)*beta;
        }
        else
        {
            result += pos(dim)*pos(dim);
        }
    }
    return -alpha*result;
}


// double wave_function_exponent_1d(double x, double alpha)
// {
//     return -alpha*x*x;
// }

#endif