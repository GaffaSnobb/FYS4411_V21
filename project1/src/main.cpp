#include "VMC.h"
#include "methods.h"

void print_parameters(
    bool parallel,
    bool interaction,
    int n_dims,
    int n_mc_cycles,
    int n_variations,
    int n_gd_iterations,
    double learning_rate,
    double initial_alpha_gd,
    double beta,
    double importance_time_step,
    double gd_tolerance,
    double brute_force_step_size,
    bool gradient_descent,
    bool importance_sampling,
    bool brute_force,
    bool numerical_differentiation
)
{
    std::cout << "PARAMETERS:" << std::endl;
    std::cout << "--------------------------" << std::endl;

    std::cout << "OpenMP: " << parallel << std::endl;
    std::cout << "Interaction: " << interaction << std::endl;
    std::cout << "gradient_descent: " << gradient_descent << std::endl;
    std::cout << "importance_sampling: " << importance_sampling << std::endl;
    std::cout << "brute_force: " << brute_force << std::endl;
    std::cout << "numerical_differentiation: " << numerical_differentiation << std::endl;
    std::cout << "n dims: " << n_dims << std::endl;

    std::cout << "n_mc_cycles: " << n_mc_cycles << std::endl;
    std::cout << "n_variations: " << n_variations << std::endl;
    std::cout << "n_gd_iterations: " << n_gd_iterations << std::endl;
    std::cout << "learning_rate: " << learning_rate << std::endl;
    std::cout << "initial_alpha_gd: " << initial_alpha_gd << std::endl;
    std::cout << "beta: " << beta << std::endl;
    std::cout << "importance_time_step: " << importance_time_step << std::endl;
    std::cout << "gd_tolerance: " << gd_tolerance << std::endl;
    std::cout << "brute_force_step_size: " << brute_force_step_size << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{   /*
    TODO: Currently omega_ho and omega_z are equal. Fix. (local_energy.cpp).
    Whats the value of omega_z?
    */

    // Parameter definitions.
    int n_mc_cycles;          // Number of MC cycles, must be a power of 2
    int n_gd_iterations;      // Max. gradient descent iterations.
    int n_variations;         // Number of variational parameters. Not in use with GD.
    double brute_force_step_size;
    double beta;
    double learning_rate;     // GD learning rate.
    double initial_alpha_gd;  // Initial variational parameter. Only for GD.
    double importance_time_step;
    bool parallel;
    arma::Col<double> alphas;

    // Global parameters:
    const int n_dims = 3;           // Number of dimensions.
    int n_particles = 10;     // Number of particles. NB: May be overwritten later in this function.
    const bool interaction = false;
    const bool debug = false;       // Toggle debug print on / off.
    double seed = 1337;       // RNG seed.
    const double gd_tolerance = 1e-3;
    const bool numerical_differentiation = true;

    // Select methods (choose one at a time):
    const bool gradient_descent = false;
    const bool importance_sampling = false;
    const bool brute_force = true;
    
    #ifdef _OPENMP
        parallel = true;
    #else
        parallel = false;
    #endif

    // 3D --------------------------------------------------------------
    if ((interaction) and (n_dims == 3) and (parallel) and (gradient_descent) and (!numerical_differentiation))
    {   /*
        Interaction ON, 3D and parallelized.
        */
        n_mc_cycles = 5e4;
        n_variations = 40;
        n_gd_iterations = 200;
        learning_rate = 1e-4;
        initial_alpha_gd = 0.2;
        importance_time_step = 0.01;
        beta = 2.82843;
        alphas = arma::linspace(0.1, 1, n_variations);
    }

    else if ((!interaction) and (n_dims == 3) and (parallel) and (gradient_descent) and (!numerical_differentiation))
    {   /*
        Interaction OFF, 3D and parallelized.
        */
        n_mc_cycles = 1e4;
        n_variations = 40;
        n_gd_iterations = 500;
        learning_rate = 1e-4;
        initial_alpha_gd = 0.2;
        importance_time_step = 0.1;
        beta = 1;
        alphas = arma::linspace(0.1, 1, n_variations);
    }

    else if ((interaction) and (n_dims == 3) and (parallel) and (importance_sampling) and (!numerical_differentiation))
    {   /*
        Interaction ON, 3D, parallelized and importance.
        */
        n_mc_cycles = 3e4;
        n_variations = 20;
        importance_time_step = 0.01;
        beta = 2.82843;
        alphas = arma::linspace(0.1, 0.6, n_variations);
    }

    else if ((interaction) and (n_dims == 3) and (!parallel) and (importance_sampling) and (!numerical_differentiation))
    {   /*
        Interaction ON, 3D, serial and importance.
        */
        n_particles = 5;     // Number of particles.
        n_mc_cycles = 1e5;
        n_variations = 20;
        importance_time_step = 0.01;
        beta = 2.82843;
        alphas = arma::linspace(0.3, 0.7, n_variations);
    }

    else if ((!interaction) and (n_dims == 3) and (!parallel) and (importance_sampling) and (!numerical_differentiation))
    {   /*
        Interaction OFF, 3D, serial and importance.
        */
        n_particles = 10;     // Number of particles.
        n_mc_cycles = 1e5;
        n_variations = 30;
        importance_time_step = 0.01;
        beta = 1;
        alphas = arma::linspace(0.4, 0.6, n_variations);
    }

    else if ((!interaction) and (n_dims == 3) and (parallel) and (importance_sampling) and (!numerical_differentiation))
    {   /*
        Interaction OFF, 3D, parallel and importance.
        */
        n_mc_cycles = 1e6;
        n_variations = 30;
        importance_time_step = 0.01;
        beta = 1;
        alphas = arma::linspace(0.1, 1, n_variations);
    }

    else if ((!interaction) and (n_dims == 3) and (!parallel) and (brute_force) and (!numerical_differentiation))
    {   /*
        Interaction OFF, 3D, serial and brute force.
        */
        n_particles = 10;     // Number of particles.
        n_mc_cycles = 1e6;
        n_variations = 30;
        beta = 1;
        alphas = arma::linspace(0.4, 0.6, n_variations);
        brute_force_step_size = 0.2;
    }

    else if ((!interaction) and (n_dims == 3) and (parallel) and (brute_force) and (!numerical_differentiation))
    {   /*
        Interaction OFF, 3D, parallel and brute force.
        */
        n_particles = 10;     // Number of particles.
        n_mc_cycles = 1e6;
        n_variations = 30;
        beta = 1;
        alphas = arma::linspace(0.4, 0.6, n_variations);
        brute_force_step_size = 0.2;
    }

    else if ((interaction) and (n_dims == 3) and (parallel) and (brute_force) and (!numerical_differentiation))
    {   /*
        Interaction ON, 3D, parallel and brute force.
        */
        n_particles = 10;     // Number of particles.
        n_mc_cycles = 1e5;
        n_variations = 20;
        beta = 2.82843;
        alphas = arma::linspace(0.1, 0.5, n_variations);
        brute_force_step_size = 0.2;
    }

    else if (!interaction and (n_dims == 3) and parallel and brute_force and numerical_differentiation)
    {   /*
        Interaction OFF, 3D, parallel, brute force and numerical
        differentiation.
        */
        n_particles = 10;
        n_mc_cycles = 1e5;
        n_variations = 20;
        beta = 1;
        alphas = arma::linspace(0.1, 1, n_variations);
        brute_force_step_size = 0.2;
    }

    // 3D end ----------------------------------------------------------
    // 2D --------------------------------------------------------------
    else if (!interaction and (n_dims == 2) and parallel and brute_force and !numerical_differentiation)
    {   /*
        Interaction OFF, 2D, parallel, brute force and analytical
        differentiation.
        */
        n_particles = 10;
        n_mc_cycles = 1e5;
        n_variations = 30;
        beta = 1;
        alphas = arma::linspace(0.1, 1, n_variations);
        brute_force_step_size = 0.2;
    }

    else if (!interaction and (n_dims == 2) and parallel and brute_force and numerical_differentiation)
    {   /*
        Interaction OFF, 2D, parallel, brute force and numerical
        differentiation.
        */
        n_particles = 10;
        n_mc_cycles = 1e5;
        n_variations = 30;
        beta = 1;
        alphas = arma::linspace(0.1, 1, n_variations);
        brute_force_step_size = 0.2;
    }
    // 2D end ----------------------------------------------------------
    // 1D --------------------------------------------------------------
    else if (!interaction and (n_dims == 1) and !parallel and brute_force and !numerical_differentiation)
    {   /*
        Interaction OFF, 1D, serial and brute force.
        */
        n_particles = 10;
        n_mc_cycles = 1e5;
        n_variations = 30;
        beta = 1;
        alphas = arma::linspace(0.1, 1, n_variations);
        brute_force_step_size = 0.2;
    }

    else if (!interaction and (n_dims == 1) and !parallel and brute_force and numerical_differentiation)
    {   /*
        Interaction OFF, 1D, serial, brute force and numerical differentiation.
        */
        n_particles = 10;
        n_mc_cycles = 1e5;
        n_variations = 10;
        beta = 1;
        alphas = arma::linspace(0.1, 1, n_variations);
        brute_force_step_size = 0.2;
    }

    else if (!interaction and (n_dims == 1) and parallel and brute_force and numerical_differentiation)
    {   /*
        Interaction OFF, 1D, parallel, brute force and numerical differentiation.
        */
        n_particles = 10;
        n_mc_cycles = 1e5;
        n_variations = 30;
        beta = 1;
        alphas = arma::linspace(0.1, 1, n_variations);
        brute_force_step_size = 0.2;
    }
    // 1D end ----------------------------------------------------------
    else
    {
        std::cout << "No parameters specified for:";
        std::cout << "\nn_dims: " << n_dims;
        std::cout << "\ninteraction: " << interaction;
        std::cout << "\nparallel: " << parallel;
        std::cout << "\ngradient_descent: " << gradient_descent;
        std::cout << "\nimportance_sampling: " << importance_sampling;
        std::cout << "\nbrute_force: " << brute_force;
        std::cout << "\nnumerical_differentiation: " << numerical_differentiation;
        std::cout << "\nExiting...\n" << std::endl;
        exit(0);
    }

    print_parameters(
        parallel,
        interaction,
        n_dims,
        n_mc_cycles,
        n_variations,
        n_gd_iterations,
        learning_rate,
        initial_alpha_gd,
        beta,
        importance_time_step,
        gd_tolerance,
        brute_force_step_size,
        gradient_descent,
        importance_sampling,
        brute_force,
        numerical_differentiation
    );
    
    #ifdef _OPENMP
        double t1 = omp_get_wtime();
        double t2;
        double comp_time;
    #else
        std::chrono::steady_clock::time_point t1;
        std::chrono::steady_clock::time_point t2;
        std::chrono::duration<double> comp_time;
        t1 = std::chrono::steady_clock::now();
    #endif

    // -----------------------------------------------------------------
    // Importance:
    if (importance_sampling)
    {
        std::cout << "Importance sampling" << std::endl;
        ImportanceSampling system_1(
            n_dims,                 // Number of spatial dimensions.
            n_variations,           // Number of variational parameters.
            n_mc_cycles,            // Number of Monte Carlo cycles.
            n_particles,            // Number of particles.
            alphas,
            beta,
            importance_time_step,
            numerical_differentiation,
            debug
        );
        system_1.set_wave_function(interaction);
        system_1.set_quantum_force(interaction);
        system_1.set_local_energy(interaction);
        system_1.set_seed(seed);
        system_1.solve();
        system_1.write_to_file_particles("generated_data/output_importance_particles.txt");
        
        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
            std::cout << "total time: " << comp_time << "s\n" << std::endl;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            std::cout << "total time: " << comp_time.count() << "s\n" << std::endl;
        #endif
    }

    // -----------------------------------------------------------------
    // Brute:
    if (brute_force)
    {
        #ifdef _OPENMP
            t1 = omp_get_wtime();
        #else
            t1 = std::chrono::steady_clock::now();
        #endif

        std::cout << "Brute force metropolis" << std::endl;

        BruteForce system_2(
            n_dims,                 // Number of spatial dimensions.
            n_variations,           // Number of variational parameters.
            n_mc_cycles,            // Number of Monte Carlo cycles.
            n_particles,            // Number of particles.
            alphas,
            beta,
            brute_force_step_size,  // Step size for new positions for brute force.
            numerical_differentiation,
            debug
        );
        system_2.set_wave_function(interaction);
        system_2.set_quantum_force(interaction);
        system_2.set_local_energy(interaction);
        system_2.set_seed(seed);
        system_2.solve();
        system_2.write_to_file_particles("generated_data/output_brute_force_particles.txt");
        
        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
            std::cout << "total time: " << comp_time << "s\n" << std::endl;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            std::cout << "total time: " << comp_time.count() << "s\n" << std::endl;
        #endif
    }

    // -----------------------------------------------------------------
    // GD:
    if (gradient_descent)
    {
        #ifdef _OPENMP
            t1 = omp_get_wtime();
        #else
            t1 = std::chrono::steady_clock::now();
        #endif

        std::cout << "Gradient decent" << std::endl;

        GradientDescent system_3(
            n_dims,                 // Number of spatial dimensions.
            n_gd_iterations,        // Number of variational parameters.
            n_mc_cycles,            // Number of Monte Carlo cycles.
            n_particles,            // Number of particles.
            importance_time_step,   // Time step size for importance sampling.
            learning_rate,          // Learning rate for GD.
            initial_alpha_gd,       // Initial guess for the variational parameter.
            beta,
            numerical_differentiation,
            debug
        );
        system_3.set_wave_function(interaction);
        system_3.set_quantum_force(interaction);
        system_3.set_local_energy(interaction);
        system_3.set_seed(seed);
        system_3.solve(gd_tolerance);
        system_3.write_to_file_particles("generated_data/output_gradient_descent_particles.txt");
        
        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
            std::cout << "total time: " << comp_time << "s\n" << std::endl;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            std::cout << "total time: " << comp_time.count() << "s\n" << std::endl;
        #endif
    }

    print_parameters(
        parallel,
        interaction,
        n_dims,
        n_mc_cycles,
        n_variations,
        n_gd_iterations,
        learning_rate,
        initial_alpha_gd,
        beta,
        importance_time_step,
        gd_tolerance,
        brute_force_step_size,
        gradient_descent,
        importance_sampling,
        brute_force,
        numerical_differentiation
    );

    return 0;
}
