//
// Created by Semigroup on 2023/10/25.
//

#ifndef NUMMKL_MATRIX_UTILS_HPP
#define NUMMKL_MATRIX_UTILS_HPP
#include "matrix_core.hpp"
#include "mkl.h"
#include <vector>

namespace nm {

void matmul(const f32 alpha, Matrix<f32> const& A, Matrix<f32> const& B, const f32 beta,
            Matrix<f32>& C, bool transA = false, bool transB = false)
{
    auto ta = transA ? CblasTrans : CblasNoTrans;
    auto tb = transB ? CblasTrans : CblasNoTrans;
    const i64 M = C.shape[0];
    const i64 N = C.shape[1];
    const i64 K = transA ? A.shape[0] : A.shape[1];
    const f32* mA = A.data;
    const i64 ldA = A.lda;
    const f32* mB = B.data;
    const i64 ldB = B.lda;
    f32* mC = C.data;
    const i64 ldC = C.lda;
    cblas_sgemm(CblasRowMajor, ta, tb, M, N, K, alpha, mA, ldA, mB, ldB, beta, mC, ldC);
}

void matmul(const f64 alpha, Matrix<f64> const& A, Matrix<f64> const& B, const f64 beta,
            Matrix<f64>& C, bool transA = false, bool transB = false)
{
    auto ta = transA ? CblasTrans : CblasNoTrans;
    auto tb = transB ? CblasTrans : CblasNoTrans;
    const i64 M = C.shape[0];
    const i64 N = C.shape[1];
    const i64 K = transA ? A.shape[0] : A.shape[1];
    const f64* mA = A.data;
    const i64 ldA = A.lda;
    const f64* mB = B.data;
    const i64 ldB = B.lda;
    f64* mC = C.data;
    const i64 ldC = C.lda;
    cblas_dgemm(CblasRowMajor, ta, tb, M, N, K, alpha, mA, ldA, mB, ldB, beta, mC, ldC);
}

f32 mean(Matrix<f32> const& mat)
{
    f32 sum = 0;
    for (int i = 0; i < mat.shape[0]; ++i) {
        sum += cblas_sasum(mat.shape[1], mat[i], 1);
    }
    f32 n = (f32)(mat.shape[0] * mat.shape[1]);
    return sum / n;
}

auto block_divide(nm::Matrix<float> const& A, int block_size)
{
    using namespace nm;
    using std::vector;
    int block_count = A.lda / block_size;
    vector<vector<Matrix<float>>> blocks;
    if (A.lda % block_size != 0 || A.shape[0] != A.shape[1]) return blocks;
    blocks.resize(block_count);
    for (int i = 0; i < block_count; i++) {
        blocks[i].reserve(block_count);
        for (int j = 0; j < block_count; j++) {
            blocks[i].emplace_back(block_size, block_size, A.lda,
                                   (A.data + i * block_size * A.lda + j * block_size));
        }
    }
    return blocks;
}

} // namespace nm

#endif // NUMMKL_MATRIX_UTILS_HPP
