#include <armadillo>

int main()
{   
    arma::vec M(3);
    M.zeros();
    double a = arma::norm(M, 2);
    std::cout << a << std::endl;
    return 0;
}