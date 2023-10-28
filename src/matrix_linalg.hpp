//
// Created by Semigroup on 2023/10/23.
//

#ifndef NUMMKL_MATRIX_LINALG_HPP
#define NUMMKL_MATRIX_LINALG_HPP

#include "matrix_core.hpp"
#include "mkl.h"

namespace nm::linalg {

template <typename T>
struct mkl_ptr_deleter
{
    void operator()(T* ptr) const
    {
        mkl_free(ptr);
        printf("mkl_free\n");
    }
};

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

inline void solve_triangular(const Matrix<float>& a, Matrix<float>& b, char side = 'l',
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

inline auto plu(Matrix<float>& mat) -> std::unique_ptr<i64[], mkl_ptr_deleter<i64>>
{
    i64 m, n, lda, l, info;
    m = mat.shape[0];
    n = mat.shape[1];
    lda = mat.lda;
    l = m < n ? m : n;
    i64* ipiv = (i64*)mkl_malloc(sizeof(i64) * l, 64);
    std::unique_ptr<i64[], mkl_ptr_deleter<i64>> ipiv_ptr(ipiv);
    info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, mat.data, lda, ipiv);
    if (info == 0) {
        return ipiv_ptr;
    }
    else if (info > 0) {
        std::cerr << "[Warning] plu: The matrix is singular\n";
    }
    else {
        std::cerr << "[Warning] plu: parameter " << -info << " is illegal\n";
    }
    mkl_free(ipiv);
    return nullptr;
}

inline void inplace_inv(Matrix<f32>& mat)
{
    i64 m, n, lda, l, info;
    m = mat.shape[0];
    n = mat.shape[1];
    lda = mat.lda;
    l = m < n ? m : n;
    i64* ipiv = (i64*)mkl_malloc(sizeof(i64) * l, 64);
    info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, mat.data, lda, ipiv);
    if (info == 0) {
        LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, mat.data, lda, ipiv);
    }
    else if (info > 0) {
        std::cerr << "[Warning] inv: The matrix is singular\n";
    }
    else {
        std::cerr << "[Warning] inv: parameter " << -info << " is illegal\n";
    }
    mkl_free(ipiv);
}

inline Matrix<f32> inv(Matrix<f32> mat)
{
    inplace_inv(mat);
    return mat;
}

inline f32 det(Matrix<f32> m)
{
    auto ipiv = plu(m);
    if (ipiv == nullptr) return std::nanf("");
    int num_swaps = 0;
    int n = m.shape[0] < m.shape[1] ? m.shape[0] : m.shape[1];
    for (int i = 0; i < n; i++) {
        if (ipiv[i] != i + 1) {
            num_swaps++;
        }
    }
    f32 determinant = (num_swaps % 2 == 0) ? 1.0f : -1.0f;
    for (int i = 0; i < n; i++) {
        determinant *= m(i, i);
    }
    return determinant;
}

} // namespace nm::linalg
#endif // NUMMKL_MATRIX_LINALG_HPP
