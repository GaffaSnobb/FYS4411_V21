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
    double importance_time_step
)
{
    std::cout << "PARAMETERS:" << std::endl;
    std::cout << "--------------------------" << std::endl;

    std::cout << "OpenMP: " << parallel << std::endl;
    std::cout << "Interaction: " << interaction << std::endl;
    std::cout << "n dims: " << n_dims << std::endl;

    std::cout << "n_mc_cycles: " << n_mc_cycles << std::endl;
    std::cout << "n_variations: " << n_variations << std::endl;
    std::cout << "n_gd_iterations: " << n_gd_iterations << std::endl;
    std::cout << "learning_rate: " << learning_rate << std::endl;
    std::cout << "initial_alpha_gd: " << initial_alpha_gd << std::endl;
    std::cout << "beta: " << beta << std::endl;
    std::cout << "importance_time_step: " << importance_time_step << std::endl;
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
    double beta;
    double learning_rate;     // GD learning rate.
    double initial_alpha_gd;  // Initial variational parameter. Only for GD.
    double importance_time_step;
    bool parallel;

    // Global parameters:
    const int n_dims = 3;           // Number of dimensions.
    const int n_particles = 10;     // Number of particles.
    const bool interaction = true;
    const bool debug = false;       // Toggle debug print on / off.
    const double seed = 1337;       // RNG seed.
    arma::Col<double> alphas;
    alphas = arma::linspace(0.1, 1, n_variations);

    // Select methods:
    const bool gradient_descent = true;
    const bool importance_sampling = false;
    const bool brute_force = false;
    

    #ifdef _OPENMP
        parallel = true;
    #else
        parallel = false;
    #endif

    if ((interaction) and (n_dims == 3) and (parallel))
    {   /*
        Interaction ON, 3D and parallelized.
        */
        n_mc_cycles = 5e4;
        n_variations = 40;
        n_gd_iterations = 50;
        learning_rate = 1e-3;
        initial_alpha_gd = 0.4;
        importance_time_step = 0.01;
        beta = 2.82843;
        // importance_time_step = 0.1; // Time step for importance serial. 0.1 - 0.01 is good.
    }

    else if ((!interaction) and (n_dims == 3) and (parallel))
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
    }

    else
    {
        std::cout << "No parameters specified for n_dims: " << n_dims;
        std::cout << ", interaction: " << interaction << " and parallel: ";
        std::cout << parallel << ". Exiting..." << std::endl;
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
        importance_time_step
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
            debug
        );
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
            const double brute_force_step_size = 0.5;   // Brute force step size parallel. 0.5 and above works.
            t1 = omp_get_wtime();
        #else
            const double brute_force_step_size = 0.2;   // Brute force step size parallel. 0.2 is good.
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
            debug
        );
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
            debug
        );
        system_3.set_wave_function(interaction);
        system_3.set_quantum_force(interaction);
        system_3.set_local_energy(interaction);
        system_3.set_seed(seed);
        system_3.solve();
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
        importance_time_step
    );

    return 0;
}
