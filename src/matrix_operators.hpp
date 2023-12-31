//
// Created by Semigroup on 2023/10/19.
//

#ifndef NUMMKL_MARTIX_OPERATORS_HPP
#define NUMMKL_MARTIX_OPERATORS_HPP
#include "matrix_core.hpp"
#include "mkl_cblas.h"
#include <iostream>

namespace nm {

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& B) const
{
    if (_shape != B.shape) {
        std::cerr << "[Warning] (+): Shape mismatch\n";
        return *this;
    }
    Matrix<T> result(shape[0], shape[1]);
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            result(i, j) = _data[i * _lda + j] + B(i, j);
        }
    }
    return result;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& B)
{
    if (_shape != B.shape) {
        std::cerr << "[Warning] (-=): Shape mismatch\n";
        return *this;
    }
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            _data[i * _lda + j] += B(i, j);
        }
    }
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& B) const
{
    if (_shape != B.shape) {
        std::cerr << "[Warning] (-): Shape mismatch\n";
        return *this;
    }
    Matrix<T> res(shape[0], shape[1]);
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            res(i, j) = _data[i * _lda + j] - B(i, j);
        }
    }
    return res;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& B)
{
    if (_shape != B.shape) {
        std::cerr << "[Warning] (-=): Shape mismatch\n";
        return *this;
    }
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            _data[i * _lda + j] -= B(i, j);
        }
    }
    return *this;
}



// For single precision
template <>
Matrix<float> Matrix<float>::operator*(const Matrix<float>& B) const
{
    if (_shape[1] != B.shape[0]) {
        std::cerr << "[Warning] (*): Shape mismatch\n";
        return *this;
    }
    Matrix<float> result(_shape[0], B.shape[1]);
    const i64 M = _shape[0];
    const i64 N = B.shape[1];
    const i64 K = _shape[1];
    const f32 alpha = 1.0;
    const f32* mA = _data;
    const i64 ld = _lda;
    const f32* mB = B._data;
    const i64 ldb = B.lda;
    const f32 beta = 0.0;
    const i64 ldc = result.lda;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                mA, ld, mB, ldb, beta, result._data, ldc);
    return result;
}

// For double precision
template <>
Matrix<double> Matrix<double>::operator*(const Matrix<double>& B) const
{
    if (_shape[1] != B.shape[0]) {
        std::cerr << "[Warning] (*): Shape mismatch\n";
        return *this;
    }
    Matrix<double> result(_shape[0], B.shape[1]);
    const i64 M = _shape[0];
    const i64 N = B.shape[1];
    const i64 K = _shape[1];
    const f64 alpha = 1.0;
    const f64* mA = _data;
    const i64 ld = _lda;
    const f64* mB = B._data;
    const i64 ldb = B.lda;
    const f64 beta = 0.0;
    const i64 ldc = result.lda;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                mA, ld, mB, ldb, beta, result._data, ldc);
    return result;
}

template<>
float Matrix<float>::min() const
{
    auto i = cblas_isamin(_shape[0] * _shape[1], _data, 1);
    return _data[i];
}

template<>
float Matrix<float>::max() const
{
    auto i = cblas_isamax(_shape[0] * _shape[1], _data, 1);
    return _data[i];
}

template <typename T>
void printMatrix(Matrix<T> const& mat) {
    int rows = mat.shape[0];
    int cols = mat.shape[1];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f\t", mat(i,j)); // Accessing matrix[i][j] using pointers
        }
        printf("\n");
    }
    printf("\n");
}

} // namespace nm

#endif
