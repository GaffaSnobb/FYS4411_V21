#include <armadillo>
#include <random>
#include "omp.h"

void not_thread_safe()
{
    arma::Mat<int> M(20, 2);
    M.zeros();
    std::mt19937 engine(1337);
    std::uniform_int_distribution<int> distribution(1, 10);
    omp_set_num_threads(2);
    omp_set_dynamic(0);

    #pragma omp parallel
    {
        std::cout << "\n" << omp_get_thread_num() << std::endl;
        for (int i = 0; i < 20; i++)
        {
            M(i, omp_get_thread_num()) = distribution(engine);
        }
    }
    M.print();
}

int main()
{   
    arma::Mat<int> M(20, 3);
    M.zeros();
    std::mt19937 engine;
    std::uniform_int_distribution<int> distribution(1, 10);
    omp_set_num_threads(3);
    omp_set_dynamic(0);

    #pragma omp parallel private(engine)
    {   
        engine.seed(omp_get_thread_num());
        std::cout << "\n" << omp_get_thread_num() << std::endl;
        for (int i = 0; i < 20; i++)
        {
            M(i, omp_get_thread_num()) = distribution(engine);
        }
    }
    M.print();
    return 0;
}