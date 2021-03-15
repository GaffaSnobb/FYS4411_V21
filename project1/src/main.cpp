#include "VMC.h"
#include "methods.h"

int main(int argc, char *argv[])
{
    const int n_mc_cycles = 1e4;//pow(2, 20);   // Number of MC cycles, must be a power of 2
    const int n_dims = 3;                 // Number of dimentions
    const int n_variations = 40;          // Number of variational parameters.
    const int gd_iterations = 100;        // Max. gradient descent iterations.
    const int n_particles = 10;           // Number of particles
    const double brute_force_step_size = 0.2;
    arma::Col<double> alphas;
    alphas = arma::linspace(0.1, 1, n_variations);

    
    #ifdef _OPENMP
        // const double importance_time_step = 0.0008;        // Time step for importance parallel.
        const double importance_time_step = 0.01;        // Time step for importance parallel.
        std::cout << "OpenMP active\n" << std::endl;
        double t1 = omp_get_wtime();
        double t2;
        double comp_time;
    #else
        const double importance_time_step = 0.1;        // Time step for importance serial. 0.1 - 0.01 is good.
        std::cout << "OpenMP inactive\n" << std::endl;
        std::chrono::steady_clock::time_point t1;
        std::chrono::steady_clock::time_point t2;
        std::chrono::duration<double> comp_time;
        t1 = std::chrono::steady_clock::now();
    #endif

    // -----------------------------------------------------------------
    // Importance:
    std::cout << "Importance sampling" << std::endl;

    ImportanceSampling system_1(
        n_dims,                 // Number of spatial dimensions.
        n_variations,           // Number of variational parameters.
        n_mc_cycles,            // Number of Monte Carlo cycles.
        n_particles,            // Number of particles.
        alphas,
        importance_time_step
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

    // // -----------------------------------------------------------------
    // // Brute:
    // #ifdef _OPENMP
    //     t1 = omp_get_wtime();
    // #else
    //     t1 = std::chrono::steady_clock::now();
    // #endif

    // std::cout << "Brute force metropolis" << std::endl;

    // BruteForce system_2(
    //     n_dims,                 // Number of spatial dimensions.
    //     n_variations,           // Number of variational parameters.
    //     n_mc_cycles,            // Number of Monte Carlo cycles.
    //     n_particles,            // Number of particles.
    //     alphas,
    //     brute_force_step_size   // Step size for new positions for brute force.
    // );
    // system_2.solve();
    // system_2.write_to_file_particles("generated_data/output_brute_force_particles.txt");
    
    // #ifdef _OPENMP
    //     t2 = omp_get_wtime();
    //     comp_time = t2 - t1;
    //     std::cout << "total time: " << comp_time << "s\n" << std::endl;
    // #else
    //     t2 = std::chrono::steady_clock::now();
    //     comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    //     std::cout << "total time: " << comp_time.count() << "s\n" << std::endl;
    // #endif

    // // -----------------------------------------------------------------
    // // GD:
    // #ifdef _OPENMP
    //     t1 = omp_get_wtime();
    // #else
    //     t1 = std::chrono::steady_clock::now();
    // #endif

    // std::cout << "Gradient decent" << std::endl;

    // GradientDescent system_3(n_dims, gd_iterations, n_mc_cycles, n_particles);
    // system_3.solve();
    // system_3.write_to_file_particles("generated_data/output_gradient_descent_particles.txt");
    
    // #ifdef _OPENMP
    //     t2 = omp_get_wtime();
    //     comp_time = t2 - t1;
    //     std::cout << "total time: " << comp_time << "s\n" << std::endl;
    // #else
    //     t2 = std::chrono::steady_clock::now();
    //     comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    //     std::cout << "total time: " << comp_time.count() << "s\n" << std::endl;
    // #endif

    return 0;
}
