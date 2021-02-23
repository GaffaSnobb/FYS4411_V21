#include "VMC.h"
#include "methods.h"

int main(int argc, char *argv[])
{

    const int n_dims = 3;
    const double dt = 0.4;

    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    std::chrono::duration<double> comp_time;
    
    t1 = std::chrono::steady_clock::now();
    
    ImportanceSampling system_1(n_dims);
    // GradientDescent system_1(n_dims);
    system_1.set_local_energy();
    system_1.set_wave_function();
    system_1.solve();
    system_1.write_to_file("generated_data/output_importance.txt");

    t2 = std::chrono::steady_clock::now();
    comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;

    t1 = std::chrono::steady_clock::now();
    
    BruteForce system_2(n_dims);
    system_2.set_local_energy();
    system_2.set_wave_function();
    system_2.solve();
    system_2.write_to_file("generated_data/output_brute_force.txt");

    t2 = std::chrono::steady_clock::now();
    comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;



    return 0;
}
