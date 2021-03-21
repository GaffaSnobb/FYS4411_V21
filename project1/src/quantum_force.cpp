#include "quantum_force.h"

arma::Mat<double> quantum_force_3d_no_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Parameters
    ----------
    pos : arma::Mat<double> reference
        Positions of all particles.
    */
    return -4*alpha*pos.col(current_particle);
}

arma::Mat<double> quantum_force_3d_interaction(
    const arma::Mat<double> &pos,
    const double alpha,
    const double beta,
    const int current_particle,
    const int n_particles
)
{   /*
    Parameters
    ----------
    pos : arma::Mat<double> reference
        Positions of all particles.
    */

    int particle;       // Particle index.
    int particle_inner; // Particle index.
    arma::Col<double> term_1(3);
    double a = 0.0043;  // Prob. not right, so fix this.
    // Term 1.
    term_1 = { // (x, y, beta*z).
        pos(0, current_particle),
        pos(1, current_particle),
        pos(2, current_particle)*beta
    };
    term_1 *= -2*alpha*std::exp(-alpha*(
        pos(0, current_particle)*pos(0, current_particle) +
        pos(1, current_particle)*pos(1, current_particle) +
        pos(2, current_particle)*pos(2, current_particle)*beta
    ));
    // Term 1 end.

    // Term 2.
    double term_2 = 1;  // Product, not sum, thus 1.
    for (particle = 0; particle < current_particle; particle++)
    {
        term_2 *= std::exp(-alpha*(
            pos(0, particle)*pos(0, particle) + 
            pos(1, particle)*pos(1, particle) + 
            pos(2, particle)*pos(2, particle)*beta
        ));
    }
    
    for (particle = current_particle + 1; particle < n_particles; particle++)
    {
        term_2 *= std::exp(-alpha*(
            pos(0, particle)*pos(0, particle) + 
            pos(1, particle)*pos(1, particle) + 
            pos(2, particle)*pos(2, particle)*beta
        ));
    }
    // Term 2 end.

    // Term 3.
    double term_3 = 0;  // Sum, not product, thus 0.
    double particle_distance;
    for (particle = 0; particle < n_particles; particle++)
    {   /*
        exp( sum( u( |r_i - r_j| ) ) )
        */
        for (particle_inner = particle + 1; particle_inner < n_particles; particle_inner++)
        {
            particle_distance =
                arma::norm(pos.col(particle) - pos.col(particle_inner), 2);
            
            if (particle_distance > a)
            {   /*
                Interaction if the particle spacing is greater than 'a'.
                NB: Not explicity stating what happens when
                particle_distance < a, since I belive it should be 0
                then (not 1 as in the wavefunction). This is in
                agreement with log(1) = 0.
                */
                term_3 += std::log(1 - a/particle_distance);
            }
        }
    }
    term_3 = std::exp(term_3);
    // Term 3 end.

    // Term 4.
    double term_4 = 1;  // Product, not sum, thus 1.
    for (particle = 0; particle < n_particles; particle++)
    {   /*
        TODO: Consider using the calculation for term_2 for this term.
        */

        term_4 *= std::exp(-alpha*(
            pos(0, particle)*pos(0, particle) + 
            pos(1, particle)*pos(1, particle) + 
            pos(2, particle)*pos(2, particle)*beta
        ));
    }
    // Term 4 end.

    // Term 5.
    double term_5 = term_3;
    // Term 5 end.

    // Term 6.
    arma::Col<double> term_6(3);
    arma::Col<double> tmp(3);
    term_6.zeros(); // Sum, not product, thus 0.
    for (particle = 0; particle < current_particle; particle++)
    {
        particle_distance =
            arma::norm(pos.col(particle) - pos.col(current_particle), 2);

        if (particle_distance > a)
        {   /*
            Interaction if the particle spacing is greater than 'a'.
            NB: Not explicity stating what happens when
            particle_distance < a, since I belive it should be 0
            then (not 1 as in the wavefunction). This is in
            agreement with log(1) = 0.
            */
            tmp = {
                pos(0, particle) - pos(0, current_particle),
                pos(1, particle) - pos(1, current_particle),
                pos(2, particle) - pos(2, current_particle)
            };
            tmp *= 1/(1 - a/particle_distance)*a*std::pow(particle_distance, -3);

            term_6 += tmp;
        }
    }
    // Term 6 end.

    double wave_function = wave_function_3d_interaction_with_loop(
        pos,
        alpha,
        beta,
        n_particles
    );

    return 2*(term_1*term_2*term_3 + term_4*term_5*term_6)/wave_function;
}