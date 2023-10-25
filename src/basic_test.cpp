#include "lu.hpp"
#include "matrix_core.hpp"
#include "matrix_linalg.hpp"
#include "matrix_operators.hpp"
#include "matrix_random.hpp"
#include <chrono>
#include <iostream>
#include "matrix_utils.hpp"

template <typename Func>
auto time_func_ms(Func lambda)
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

    const int N = 4096;
    auto A = randMatrix<float>(N, N);
    // make A strict diagonally dominant
    for (int i = 0; i < N; ++i) {
        float sum = 0;
        for (int j = 0; j < N; ++j) {
            sum += std::abs(A(i, j));
        }
        A(i, i) = sum;
    }

    auto B = A;
    auto C = A;
    auto D = A;
    auto E = A;
    i64 t_ms;

//    t_ms = time_func_ms([&]() { sequential_lu(A.shape[0], A.data); });
//    print("sequential took", t_ms, "milliseconds.");
//    print((A[50][50]));

//    t_ms = time_func_ms([&]() { avx_lu(B.shape[0], B.data); });
//    print("avx took", t_ms, "milliseconds.");

    t_ms = time_func_ms([&]() { block_lu(C, 32); });
    print("block lu took", t_ms, "milliseconds.");

    t_ms = time_func_ms([&]() { block_lu_std_thread(E, 32); });
    print("block lu thread took", t_ms, "milliseconds.");

    t_ms = time_func_ms([&]() { linalg::lu(D); });
    print("mkl took", t_ms, "milliseconds.");

    auto m = D - C;
    auto res = linalg::norm(m, 'f');
    print("Frobenius norm of difference is", res);
    print("max difference is", m.max());
    print("mean of difference is", mean(m));
    print("END OF PROGRAM");
    return 0;
}