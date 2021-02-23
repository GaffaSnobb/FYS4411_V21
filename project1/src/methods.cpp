#include "methods.h"

void BruteForce::set_initial_positions(int dim, int particle)
{   /*
    Set the initial positions according to the brute force approach,
    drawing random numbers uniformly in the interval [-1/2, 1/2].

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.
    */
    pos_current(dim, particle) = step_size*(uniform(engine) - 0.5);
}

void BruteForce::set_new_positions(int dim, int particle)
{   /*
    Set new positions according to the brute force approach, drawing
    random numbers uniformly in the interval [-1/2, 1/2].

    Parameters
    ----------
    dim : integer
        Current dimension index.

    particle : integer
        Current particle index.
    */
    pos_new(dim, particle) = pos_current(dim, particle) + step_size*(uniform(engine) - 0.5);
}