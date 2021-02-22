#include "VMC.h"


int main()
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    const int n_dims = 3;
    // const int method = 0;
    const std::string method = "brute_force";
    const double dt = 0.4;

    VMC system(n_dims, method);

    system.set_local_energy();
    system.set_wave_function();

    if (method == "brute_force")
    {
      system.brute_force();
      system.write_to_file("generated_data/output_bruteforce.txt");
    }
    else if (method == "importance_sampling")
    {
      system.importance_sampling(dt);
      system.write_to_file("generated_data/output_importance.txt");
    }
    else
    {
      std::cout << "no method chosen, brute_force:0 or importance:1"<<std::endl;
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> comp_time = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);

    std::cout << "\ntotal time: " << comp_time.count() << "s" << std::endl;

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

    // std::string str1 = "lol";
    // std::string str2 = "lol";
    
    // if (str1 == "lol")
    // {
    //     std::cout << "lolz" << std::endl;
    // }




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
