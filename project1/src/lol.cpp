#include "omp.h"
#include <armadillo>
#include <iostream>

class SubClass
{
public:
    arma::Col<double> M;
    int end = 10;
    SubClass(int a) // HVORFOR MÅ JEG HA ARGUMENT FOR Å KALLE PÅ KONSTRUKTØREN?
    {
        M = arma::Col<double>(end);
    }
    
    void virtual_method()
    {   
        std::cout << "non_virtual_method" << std::endl;
        #pragma omp parallel for
        for (int i = 0; i < end; i++)
        {
            std::cout << "\n" << omp_get_thread_num() << std::endl;
            M(i) = i;

        }
    }
};

int main()
{   
    // #pragma omp parallel for
    // for (int i = 0; i < 5; i++)
    // {
    //     std::cout << "thread: " << omp_get_thread_num() << std::endl;
    // }
    // std::cout << "LOLZ ute" << std::endl;
    // SubClass q(2);
    // q.virtual_method();
    int var_1 = 0, var_2 = 0;
    int N = 10;
    arma::Col<int> M(N);
    M.zeros();
    M(3) = 88;

    #pragma omp parallel for reduction(+:var_1) \
        firstprivate(M)
    for (int i = 1; i < N; i++)
    {
        var_1 = i;
        var_2 = i;

        if (omp_get_thread_num() == 0)
        {
            M.print();
        }
    }

    std::cout << "var_1: " << var_1 << std::endl;
    std::cout << "var_2: " << var_2 << std::endl;
    return 0;
}