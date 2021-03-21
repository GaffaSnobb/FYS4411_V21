#include <iostream>
#include "omp.h"

class MyClass
{
    public:
        MyClass(int a)
        {   
            int i;
            std::cout << "LOLZ constructror" << std::endl;
            // #pragma omp parallel for private(a, i)
            // for (i = 0; i < 5; i++)
            // {
            //     std::cout << "\n" << omp_get_thread_num() << std::endl;
            // }
            // virtual_method();
        }
        void method_2()
        {
            virtual_method();
        }
    protected:
        virtual void virtual_method()
        {
            // std::cout << "virtual_method" << std::endl;
        }
};