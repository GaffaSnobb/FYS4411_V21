#include "VMC.h"


int main()
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    const int n_dims = 3;

    VMC system(n_dims);
    system.set_local_energy();
    system.set_wave_function();
    system.brute_force();
    system.write_to_file("generated_data/output_bruteforce.txt");

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // -------------------------------
    // Importance sampling starts here

    double dt = 0.4;  // Set the time step 0.4 is good

    VMC system2(n_dims);
    system2.set_local_energy();
    system2.set_wave_function();
    system2.importance_sampling(dt);
    system2.write_to_file("generated_data/output_importance.txt");

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> comp_time1 = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3);

    std::cout << "\ntotal time: " << comp_time1.count() << "s" << std::endl;

    return 0;
}
