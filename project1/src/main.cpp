#include "VMC.h"
#include "methods.h"

int main(int argc, char *argv[])
{

    const int n_dims = 3;
    const double dt = 0.4;
    // const int method = 0;
    // const std::string method = "brute_force";
    // const std::string method = "importance_sampling";

    std::chrono::steady_clock::time_point t1;
    t1 = std::chrono::steady_clock::now();
    // VMC system(n_dims);
    
    // BruteForce system(n_dims);
    ImportanceSampling system(n_dims);

    // if (method == "brute_force")
    // {
    //     BruteForce system(n_dims);
    // }
    // else if (method == "importance_sampling")
    // {
    //     ImportanceSampling system(n_dims);
    // }
    // else
    // {
    //     VMC system(n_dims);
    // }

    system.set_local_energy();
    system.set_wave_function();
    system.solve();
    // system.write_to_file("generated_data/output_brute_force.txt");
    system.write_to_file("generated_data/output_importance.txt");

    std::chrono::steady_clock::time_point t2;
    t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> comp_time;
    comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();


    // -------------------------------
    // Importance sampling starts here

    //double dt = 0.4;  // Set the time step 0.4 is good
    //const int method_input_i = 1;

    //VMC system2(n_dims_input, method_input_i);
    //system2.set_local_energy();
    //system2.set_wave_function();
    //system2.importance_sampling(dt);
    //system2.write_to_file("generated_data/output_importance.txt");

    //std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    //std::chrono::duration<double> comp_time1 = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3);

    //std::cout << "\ntotal time: " << comp_time1.count() << "s" << std::endl;

    return 0;
}
