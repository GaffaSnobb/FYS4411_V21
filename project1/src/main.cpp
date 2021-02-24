#include "VMC.h"
#include "methods.h"
#include<cmath>
int main(int argc, char *argv[])
{
    const int n_dims = 3;                 // Number of dimentions
    const int n_variations = pow(2, 7);   // Number of variational parameters.
    const int gd_iterations = 100;        // Max. gradient descent iterations.
    const int n_mc_cycles = 1e3;
    const int n_particles = 10;
    // const double time_step = 0.4;      // Time step for importance. TODO: Make this an input to ImportanceSampling.

    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    std::chrono::duration<double> comp_time;

    // Importance:
    t1 = std::chrono::steady_clock::now();

    ImportanceSampling system_1(n_dims, n_variations, n_mc_cycles, n_particles);
    system_1.solve();
    system_1.write_to_file("generated_data/output_importance.txt");

    t2 = std::chrono::steady_clock::now();
    comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;

    // Brute:
    t1 = std::chrono::steady_clock::now();

    BruteForce system_2(n_dims, n_variations, n_mc_cycles, n_particles);
    system_2.solve();
    system_2.write_to_file("generated_data/output_brute_force.txt");

    t2 = std::chrono::steady_clock::now();
    comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;

    // GD:
    t1 = std::chrono::steady_clock::now();

    GradientDescent system_3(n_dims, gd_iterations, n_mc_cycles, n_particles);
    system_3.solve();
    system_3.write_to_file("generated_data/output_gradient_descent.txt");

    t2 = std::chrono::steady_clock::now();
    comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;

    return 0;
}
