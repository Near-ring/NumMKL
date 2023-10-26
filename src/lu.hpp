//
// Created by Semigroup on 2023/10/23.
//

#ifndef NUMMKL_LU_HPP
#define NUMMKL_LU_HPP
#include "avx_utils.hpp"
#include "matrix_core.hpp"
#include "matrix_linalg.hpp"
#include "matrix_utils.hpp"
#include <thread>
#include <vector>
#include <omp.h>

void sequential_lu(int n, float* A)
{
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

inline void avx_lu(int n, float* A)
{
    const int ub = n - 8;
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
            int j;
            for (j = k + 1; j <= ub; j += 8) {
                auto l_ik = _mm256_broadcast_ss(&A[i * n + k]);
                auto subtrahend = _mm256_mul_ps(l_ik, _mm256_load_ps(&A[k * n + j]));
                nm::sub8f32_ps(&A[i * n + j], subtrahend, &A[i * n + j]);
            }
            for (; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

inline void avx_lu_m(nm::Matrix<float>& a)
{
    const int n = a.shape[0];
    const int ub = n - 8;
    float* A = a.data;
    const int lda = a.lda;
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            A[i * lda + k] /= A[k * lda + k];
            int j;
            for (j = k + 1; j <= ub; j += 8) {
                auto l_ik = _mm256_broadcast_ss(&A[i * lda + k]);
                auto subtrahend = _mm256_mul_ps(l_ik, _mm256_load_ps(&A[k * lda + j]));
                nm::sub8f32_ps(&A[i * lda + j], subtrahend, &A[i * lda + j]);
            }
            for (; j < n; j++) {
                A[i * lda + j] -= A[i * lda + k] * A[k * lda + j];
            }
        }
    }
}


void block_lu(nm::Matrix<float>& A, int block_size)
{
    using namespace nm;
    bool valid = true;
    valid &= (A.shape[0] == A.shape[1] && A.shape[1] == A.lda);
    valid &= (block_size <= A.lda && A.shape[1] % block_size == 0);

    if (!valid) {
        printf("\n[WARNING] block_lu: invalid block size or matrix shape.\n\n");
        return;
    }

    auto block = block_divide(A, block_size);
    int block_count = A.lda / block_size;

    // omp_set_num_threads(4);
    // iterate diagonal blocks
    for (int i = 0; i < block_count; i++) {
        //linalg::lu(block[i][i]);
        avx_lu_m(block[i][i]);

        // update the blocks to the right
        for (int j = i + 1; j < block_count; j++) {
            linalg::solve_triangular(block[i][i], block[i][j], 'l', true, true);
        }

        // update the blocks below
        for (int j = i + 1; j < block_count; j++) {
            linalg::solve_triangular(block[i][i], block[j][i], 'r', false, false);
        }

        // update matrix in south-east corner
        for (int ii = i + 1; ii < block_count; ii++) {
            for (int jj = i + 1; jj < block_count; jj++) {
                block[ii][jj] -= block[ii][i] * block[i][jj];
                //block[ii][jj] = block[ii][jj] - block[ii][i] * block[i][jj];
                //matmul(-1.0, block[ii][i], block[i][jj], 1.0, block[ii][jj]);
            }
        }
    }
    //omp_set_num_threads(8);
}

void block_lu_std_thread(nm::Matrix<float>& A, int block_size)
{
    using namespace nm;
    bool valid = true;
    valid &= (A.shape[0] == A.shape[1] && A.shape[1] == A.lda);
    valid &= (block_size <= A.lda && A.shape[1] % block_size == 0);

    if (!valid) {
        printf("\n[WARNING] block_lu: invalid block size or matrix shape.\n\n");
        return;
    }

    auto block = block_divide(A, block_size);
    int block_count = A.lda / block_size;

    // iterate diagonal blocks
    for (int i = 0; i < block_count; i++) {
        //linalg::lu(block[i][i]);
        avx_lu_m(block[i][i]);

        // update the block to the right of the current diagonal

        std::thread t1([&]() {
            for (int j = i + 1; j < block_count; j++) {
                linalg::solve_triangular(block[i][i], block[i][j], 'l', true, true);
            }
        });

        std::thread t2([&]() {
            // update the block below current diagonal
            for (int j = i + 1; j < block_count; j++) {
                linalg::solve_triangular(block[i][i], block[j][i], 'r', false, false);
            }
        });

        t1.join();
        t2.join();

        // update matrix in south-east corner
        for (int ii = i + 1; ii < block_count; ii++) {
            for (int jj = i + 1; jj < block_count; jj++) {
                //                auto t = block[ii][i] * block[i][jj];
                //                auto v = block[ii][jj] - t;
                //                block[ii][jj] = v;
                matmul(-1.0, block[ii][i], block[i][jj], 1.0, block[ii][jj]);
            }
        }
    }
}

#endif // NUMMKL_LU_HPP
