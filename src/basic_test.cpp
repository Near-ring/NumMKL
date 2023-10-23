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

void print() {
    std::cout << "\n";  // print a newline when we reach the end
}

template <typename T, typename... Args>
void print(const T& str, const Args&... args) {
    std::cout << str << " ";
    print(args...);
}

int main()
{
    using namespace nm;
    using namespace std::chrono;

    const int N = 2048;
    Matrix<float> A = randMatrix<float>(N, N);
    // make A strict diagonally dominant
    for (int i = 0; i < N; ++i) {
        float sum = 0;
        for (int j = 0; j < N; ++j) {
            sum += std::abs(A(i, j));
        }
        A(i * N + i) = sum;
    }
    auto C = A;
    auto D = A;
    auto E = A;

    auto t_ms = time_func_ms([&]() { sequential_lu(A.shape[0], A.data); });
    print("sequential took", t_ms, "milliseconds.");
    print((A[50][50]));

    t_ms = time_func_ms([&]() { avx_lu(C.shape[0], C.data); });
    print("avx took", t_ms, "milliseconds.");
    print((C[50][50]));

    t_ms = time_func_ms([&]() { omp_lu(D.shape[0], D.data); });
    print("omp took", t_ms, "milliseconds.");
    print((D[50][50]));

    t_ms = time_func_ms([&]() { linalg::lu(E); });
    print("mkl took", t_ms, "milliseconds.");
    print((E[50][50]));

    auto m = E - C;
    print(m[50][50]);
    float res = linalg::norm(m, 'f');
    print("norm of difference is", res);
    print("max difference is", m.max());
    return 0;
}