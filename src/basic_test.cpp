#include "cpp_utils.hpp"
#include "lu.hpp"
#include "num_mkl.hpp"

int main()
{
    using namespace nm;
    Matrix<float> A = {{1, 2, 3}, {3, 2, 1}, {2, 1, 3}};
//    auto B = linalg::inv(A);
//    printMatrix(B);
    f32 d = linalg::det(A);
    print(d);
}

int _main()
{
    using namespace nm;

    const int N = 2048;
    auto const A = randMatrix<float>(N, N);
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

    //    t_ms = measure_exec_time_ms([&]() { sequential_lu(A.shape[0], A.data); });
    //    print("sequential took", t_ms, "milliseconds.");

    t_ms = measure_exec_time_ms([&]() { avx_lu(B.shape[0], B.data); });
    print("avx took", t_ms, "milliseconds.");

    t_ms = measure_exec_time_ms([&]() { block_lu(C, 256); });
    print("block lu took", t_ms, "milliseconds.");

    //    t_ms = measure_exec_time_ms([&]() { block_lu_omp_thread(E, 32); });
    //    print("block lu omp thread took", t_ms, "milliseconds.");

    t_ms = measure_exec_time_ms([&]() { block_lu_std_thread(E, 512); });
    print("block lu std thread took", t_ms, "milliseconds.");

    t_ms = measure_exec_time_ms([&]() { linalg::lu(D); });
    print("mkl took", t_ms, "milliseconds.");

    auto l_u = lu_extract(C);
    auto a_lu = l_u[0] * l_u[1];
    auto m_diff = A - a_lu;
    float res = linalg::norm(m_diff, 'f');
    print("Frobenius norm of difference is", res);
    print("max difference is", m_diff.max());
    print("mean of difference is", mean(m_diff));
    print("END OF PROGRAM");
    return 0;
}