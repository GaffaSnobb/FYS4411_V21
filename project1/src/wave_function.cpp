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
        {
            result += pos(dim)*pos(dim)*beta;
        }
        else
        {
            result += pos(dim)*pos(dim);
        }
    }
    return -alpha*result;

    // double result = pos(0)*pos(0);  // At least one spatial dimension.
    // if (n_dims == 1)
    // {
    //     return -alpha*result;
    // }
    // else if (n_dims == 2)
    // {   
    //     result += pos(1)*pos(1);
    //     return -alpha*result;
    // }
    // else
    // {   
    //     result += pos(1)*pos(1) + pos(2)*pos(2);
    //     return -alpha*result;
    // }

    // switch (n_dims)
    // {
    //     case 1:
    //         return -alpha*result;
        
    //     case 2:
    //         result += pos(1)*pos(1);
    //         return -alpha*result;
        
    //     default:
    //         result += pos(1)*pos(1) + pos(2)*pos(2);
    //         return -alpha*result;
    // }
}

#endif