#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main() {
    mat X(4, 5, fill::randu);

    mat U;
    vec s;
    mat V;

    svd_econ(U, s, V, X);
    cout << U << endl;
    cout << s << endl;
    cout << V << endl;

    return 0;
}
