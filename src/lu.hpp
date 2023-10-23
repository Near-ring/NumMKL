//
// Created by Semigroup on 2023/10/23.
//

#ifndef NUMMKL_LU_HPP
#define NUMMKL_LU_HPP
#include "avx_utils.hpp"
#include "omp.h"

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

void avx_lu(int n, float* A)
{
    const int ub = n - 8;
    for (int k = 0; k <= n; k++) {
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];
            int j;
            for (j = k + 1; j <= ub; j += 8) {
                auto pivot_l = _mm256_broadcast_ss(&A[i * n + k]);
                auto subtrahend = _mm256_mul_ps(pivot_l, _mm256_load_ps(&A[k * n + j]));
                sub8f32_ps(&A[i * n + j], subtrahend, &A[i * n + j]);
            }
            for (; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

#endif // NUMMKL_LU_HPP
