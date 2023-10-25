//
// Created by Semigroup on 2023/10/23.
//

#ifndef NUMMKL_MATRIX_LINALG_HPP
#define NUMMKL_MATRIX_LINALG_HPP

#include "matrix_core.hpp"
#include "mkl.h"

namespace nm::linalg {
inline void lu(Matrix<float>& mat)
{
    i64 m, n, lda;
    m = mat.shape[0];
    n = mat.shape[1];
    lda = mat.lda;
    LAPACKE_mkl_sgetrfnp(LAPACK_ROW_MAJOR, m, n, mat.data, lda);
}

inline float norm(Matrix<float>& mat, char norm_type)
{
    i64 m, n, lda;
    m = mat.shape[0];
    n = mat.shape[1];
    lda = mat.lda;
    return LAPACKE_slange(LAPACK_ROW_MAJOR, norm_type, m, n, mat.data, lda);
}

inline void solve_triangular(Matrix<float>& a, Matrix<float>& b, char side = 'l',
                             bool lower = false, bool unit_diagonal = false)
{
    auto sd = side == 'l' ? CblasLeft : CblasRight;
    auto ul = lower ? CblasLower : CblasUpper;
    auto dg = unit_diagonal ? CblasUnit : CblasNonUnit;
    const i64 M = b.shape[0];
    const i64 N = b.shape[1];
    const f32 alpha = 1.0;
    const f32* A = a.data;
    const i64 lda = a.lda;
    f32* B = b.data;
    const i64 ldb = b.lda;
    cblas_strsm(CblasRowMajor, sd, ul, CblasNoTrans, dg, M, N, alpha, A, lda, B, ldb);
}

} // namespace nm::linalg

#endif // NUMMKL_MATRIX_LINALG_HPP
