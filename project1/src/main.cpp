#include "VMC.h"
#include "methods.h"
#include "parameters.h"

void print_parameters(
    bool parallel,
    bool interaction,
    int n_dims,
    int n_mc_cycles,
    int n_variations,
    int n_gd_iterations,
    int n_particles,
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
    std::cout << "n_particles: " << n_particles << std::endl;
    std::cout << "learning_rate: " << learning_rate << std::endl;
    std::cout << "initial_alpha_gd: " << initial_alpha_gd << std::endl;
    std::cout << "beta: " << beta << std::endl;
    std::cout << "importance_time_step: " << importance_time_step << std::endl;
    std::cout << "gd_tolerance: " << gd_tolerance << std::endl;
    std::cout << "brute_force_step_size: " << brute_force_step_size << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << std::endl;
}

void generate_filenames(
    std::string method,
    std::string &fname_particles,
    std::string &fname_onebody,
    std::string &fname_energies,
    int n_particles,
    int n_dims,
    int n_mc_cycles,
    double importance_or_brute_step,
    bool numerical_differentiation,
    bool interaction
)
{   /*
    Generate file names for particles, onebody and energy output files.
    */
    fname_particles = "generated_data/output_";
    fname_particles += method;
    fname_particles += "_";
    fname_particles += std::to_string(n_particles);
    fname_particles += "_";
    fname_particles += std::to_string(n_dims);
    fname_particles += "_";
    fname_particles += std::to_string(n_mc_cycles);
    fname_particles += "_";
    fname_particles += std::to_string(importance_or_brute_step);
    fname_particles += "_";

    if (numerical_differentiation) {fname_particles += "numerical";}
    else {fname_particles += "analytical";}
    fname_particles += "_";
    if (interaction) {fname_particles += "interaction";}
    else {fname_particles += "nointeraction";}
    fname_particles += "_";

    fname_onebody = fname_particles;
    fname_onebody += "onebody_";
    fname_onebody += std::to_string(a);
    fname_onebody += "_.txt";

    fname_energies = fname_particles;
    fname_energies += "energies_";
    fname_energies += std::to_string(a);
    fname_energies += "_.txt";

    fname_particles += "particles_";
    fname_particles += std::to_string(a);
    fname_particles += "_.txt";
}

int main(int argc, char *argv[])
{   /*

    const double importance_time_step = 0.04; funker best med mange
    partikler.

    litt over 0.28
    */
    // Parameter definitions.
    bool parallel;
    double beta;

    // Global parameters:
    double brute_force_step_size      = 0.2;
    const double importance_time_step = 0.1;
    const double initial_alpha_gd     = 0.2;                // Initial variational parameter. Only for GD.
    const double learning_rate        = 1e-4;               // GD learning rate.
    const int n_gd_iterations         = 200;                // Max. gradient descent iterations.
    long seed                         = time(NULL);
    const double gd_tolerance         = 1e-4;
    const bool debug                  = true;               // Toggle debug print on / off.

    const bool interaction               = false;
    const bool numerical_differentiation = false;
    const int n_variations               = 1;               // Number of variational parameters. Not in use with GD.
    const int n_mc_cycles                = std::pow(2, 20); // Number of MC cycles, must be a power of 2
    const int n_dims                     = 3;               // Number of dimensions.
    const int n_particles                = 10;              // Number of particles.
    arma::Col<double> alphas             = arma::linspace(0.5, 0.5, n_variations);

    // Select methods (choose one at a time):
    const bool gradient_descent    = false;
    const bool importance_sampling = true;
    const bool brute_force         = false;

    if (interaction)
    {
        beta = 2.82843;
    }
    else
    {
        beta = 1;
    }

    if ((gradient_descent and importance_sampling) or (gradient_descent and brute_force) or (importance_sampling and brute_force))
    {
        std::cout << "Please choose only one method at a time! Exiting..." << std::endl;
        exit(0);
    }
    if (!gradient_descent and !brute_force and !importance_sampling)
    {
        std::cout << "No method is chosen. Exiting..." << std::endl;
        exit(0);
    }

    #ifdef _OPENMP
        parallel = true;
    #else
        parallel = false;
    #endif

    print_parameters(
        parallel,
        interaction,
        n_dims,
        n_mc_cycles,
        n_variations,
        n_gd_iterations,
        n_particles,
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

    // Importance ------------------------------------------------------
    if (importance_sampling)
    {
        std::cout << "Importance sampling" << std::endl;

        std::string fname_importance_particles;
        std::string fname_importance_onebody;
        std::string fname_importance_energies;

        generate_filenames(
            "importance",
            fname_importance_particles,
            fname_importance_onebody,
            fname_importance_energies,
            n_particles,
            n_dims,
            n_mc_cycles,
            importance_time_step,
            numerical_differentiation,
            interaction
        );

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

        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
            std::cout << "total time: " << comp_time << "s\n" << std::endl;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            std::cout << "total time: " << comp_time.count() << "s\n" << std::endl;
        #endif
        system_1.write_to_file(fname_importance_particles);
        system_1.write_energies_to_file(fname_importance_energies);
        system_1.write_to_file_onebody_density(fname_importance_onebody);
    }

    // Brute -----------------------------------------------------------
    if (brute_force)
    {
        #ifdef _OPENMP
            t1 = omp_get_wtime();
        #else
            t1 = std::chrono::steady_clock::now();
        #endif

        std::cout << "Brute force metropolis" << std::endl;

        std::string fname_brute_particles;
        std::string fname_brute_onebody;
        std::string fname_brute_energies;

        generate_filenames(
            "brute",
            fname_brute_particles,
            fname_brute_onebody,
            fname_brute_energies,
            n_particles,
            n_dims,
            n_mc_cycles,
            brute_force_step_size,
            numerical_differentiation,
            interaction
        );

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

        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
            std::cout << "total time: " << comp_time << "s\n" << std::endl;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            std::cout << "total time: " << comp_time.count() << "s\n" << std::endl;
        #endif
        system_2.write_to_file(fname_brute_particles);
        system_2.write_energies_to_file(fname_brute_energies);
        system_2.write_to_file_onebody_density(fname_brute_onebody);
    }

    // GD --------------------------------------------------------------
    if (gradient_descent)
    {
        #ifdef _OPENMP
            t1 = omp_get_wtime();
        #else
            t1 = std::chrono::steady_clock::now();
        #endif

        std::cout << "Gradient decent" << std::endl;

        std::string fname_gradient_particles;
        std::string fname_gradient_onebody;
        std::string fname_gradient_energies;

        generate_filenames(
            "gradient",
            fname_gradient_particles,
            fname_gradient_onebody,
            fname_gradient_energies,
            n_particles,
            n_dims,
            n_mc_cycles,
            importance_time_step,
            numerical_differentiation,
            interaction
        );

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

        #ifdef _OPENMP
            t2 = omp_get_wtime();
            comp_time = t2 - t1;
            std::cout << "total time: " << comp_time << "s\n" << std::endl;
        #else
            t2 = std::chrono::steady_clock::now();
            comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
            std::cout << "total time: " << comp_time.count() << "s\n" << std::endl;
        #endif

        system_3.write_to_file(fname_gradient_particles);
        system_3.write_energies_to_file(fname_gradient_energies);
        system_3.write_to_file_onebody_density(fname_gradient_onebody);
    }

    print_parameters(
        parallel,
        interaction,
        n_dims,
        n_mc_cycles,
        n_variations,
        n_gd_iterations,
        n_particles,
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
