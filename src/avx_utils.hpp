//
// Created by Semigroup on 2023/10/21.
//

#pragma once
#include <immintrin.h>

typedef long long i64;

inline void avx_copy_f32(const float* A, float* B, size_t l)
{
    const i64 ub = l - 8;
    int i;
    for (i = 0; i <= ub; i += 8) {
        __m256 v8f_A = _mm256_loadu_ps(&A[i]);
        _mm256_storeu_ps(&B[i], v8f_A);
    }

    for (; i < l; i++) {
        B[i] = A[i];
    }
}

inline void avx_add_f32(const float* A, const float* B, float* C, size_t l)
{
    const i64 ub = l - 8;
    int i;
    for (i = 0; i <= ub; i += 8) {
        __m256 v8f_A = _mm256_loadu_ps(&A[i]);
        __m256 v8f_B = _mm256_loadu_ps(&B[i]);
        __m256 v8f_C = _mm256_add_ps(v8f_A, v8f_B);
        _mm256_storeu_ps(&C[i], v8f_C);
    }

    for (; i < l; i++) {
        C[i] = A[i] + B[i];
    }
}

inline void avx_sub_f32(const float* A, const float* B, float* C, size_t l)
{
    const i64 ub = l - 8;
    int i;
    for (i = 0; i <= ub; i += 8) {
        __m256 v8f_A = _mm256_loadu_ps(&A[i]);
        __m256 v8f_B = _mm256_loadu_ps(&B[i]);
        __m256 v8f_C = _mm256_sub_ps(v8f_A, v8f_B);
        _mm256_storeu_ps(&C[i], v8f_C);
    }

    for (; i < l; i++) {
        C[i] = A[i] - B[i];
    }
}

inline void avx_mul_f32(const float* A, const float* B, float* C, size_t l)
{
    const i64 ub = l - 8;
    int i;
    for (i = 0; i <= ub; i += 8) {
        __m256 v8f_A = _mm256_loadu_ps(&A[i]);
        __m256 v8f_B = _mm256_loadu_ps(&B[i]);
        __m256 v8f_C = _mm256_mul_ps(v8f_A, v8f_B);
        _mm256_storeu_ps(&C[i], v8f_C);
    }

    for (; i < l; i++) {
        C[i] = A[i] * B[i];
    }
}

inline void avx_div_f32(const float* A, const float* B, float* C, size_t l)
{
    const i64 ub = l - 8;
    int i;
    for (i = 0; i <= ub; i += 8) {
        __m256 v8f_A = _mm256_loadu_ps(&A[i]);
        __m256 v8f_B = _mm256_loadu_ps(&B[i]);
        __m256 v8f_C = _mm256_div_ps(v8f_A, v8f_B);
        _mm256_storeu_ps(&C[i], v8f_C);
    }

    for (; i < l; i++) {
        C[i] = A[i] / B[i];
    }
}

inline void div8f32_ps(const float* src, __m256 divisor, float* dst)
{
    __m256 v8f32_A = _mm256_load_ps(src);
    __m256 v8f32_C = _mm256_div_ps(v8f32_A, divisor);
    _mm256_store_ps(dst, v8f32_C);
}

inline void sub8f32_ps(const float* src, __m256 subtrahend, float* dst)
{
    __m256 v8f32_A = _mm256_load_ps(src);
    __m256 v8f32_C = _mm256_sub_ps(v8f32_A, subtrahend);
    _mm256_store_ps(dst, v8f32_C);
}