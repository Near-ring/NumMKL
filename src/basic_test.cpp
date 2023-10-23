#include "lu.hpp"
#include "matrix_core.hpp"
#include "matrix_linalg.hpp"
#include "matrix_operators.hpp"
#include "matrix_random.hpp"
#include <chrono>
#include <iostream>

typedef long long i64;

template <typename Func>
auto time_func_ms(Func lambda) -> i64
{
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    lambda();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    return duration.count();
}

int main()
{
    using namespace nm;
    using namespace std::chrono;

    const int N = 2400;
    Matrix<float> A = randMatrix<float>(N, N);
    Matrix<float> B = randMatrix<float>(N, N);
    auto C = A;
    auto D = A;
    auto E = A;

    auto t_ms = time_func_ms([&]() { sequential_lu(A.shape[0], A.data); });
    std::cout << "sq took " << t_ms << " milliseconds."
              << "\n";
    std::cout << (double)(A[101][204]) << "\n";

    t_ms = time_func_ms([&]() { avx_lu(C.shape[0], C.data); });
    std::cout << "avx took " << t_ms << " milliseconds."
              << "\n";
    std::cout << (double)(C[101][204]) << "\n";

    t_ms = time_func_ms([&]() { omp_lu(D.shape[0], D.data); });
    std::cout << "omp took " << t_ms << " milliseconds."
              << "\n";
    std::cout << (double)(C[101][204]) << "\n";

    t_ms = time_func_ms([&]() { linalg::lu(E); });
    std::cout << "omp took " << t_ms << " milliseconds."
              << "\n";
    std::cout << (double)(E[101][204]) << "\n";

    return 0;
}