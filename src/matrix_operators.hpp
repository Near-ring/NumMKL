//
// Created by Semigroup on 2023/10/19.
//

#ifndef NUMMKL_MARTIX_OPERATORS_HPP
#define NUMMKL_MARTIX_OPERATORS_HPP
#include "matrix_core.hpp"
#include "mkl_cblas.h"
#include <cstdio>

namespace nm {

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& B) const
{
    if (_shape != B.shape) {
        return Matrix<T>(0, 0, nullptr);
    }
    Matrix<T> result(shape[0], shape[1]);
    const int l = shape[0] * shape[1];
    for (int i = 0; i < l; ++i) {
        result._data[i] = _data[i] + B._data[i];
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& B) const
{
    if (_shape != B.shape) {
        return Matrix<T>(0, 0, nullptr);
    }
    Matrix<T> result(shape[0], shape[1]);
    const int l = shape[0] * shape[1];
    for (int i = 0; i < l; ++i) {
        result._data[i] = _data[i] - B._data[i];
    }
    return result;
}

// For single precision
template <>
Matrix<float> Matrix<float>::operator*(const Matrix<float>& B) const
{
    if (_shape[1] != B.shape[0]) {
        return {0, 0, nullptr};
    }
    Matrix<float> result(_shape[0], B.shape[1]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _shape[0], B.shape[1], _shape[1], 1.0,
                _data, _shape[1], B._data, B.shape[1], 0.0, result._data, B.shape[1]);
    return result;
}

// For double precision
template <>
Matrix<double> Matrix<double>::operator*(const Matrix<double>& B) const
{
    if (_shape[1] != B.shape[0]) {
        return {0, 0, nullptr};
    }
    Matrix<double> result(_shape[0], B.shape[1]);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _shape[0], B.shape[1], _shape[1], 1.0,
                _data, _shape[1], B._data, B.shape[1], 0.0, result._data, B.shape[1]);
    return result;
}

template <typename T>
void printMatrix(Matrix<T> const& mat) {
    auto matrix = mat.data;
    int rows = mat.shape[0];
    int cols = mat.shape[1];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f\t", mat[i][j]); // Accessing matrix[i][j] using pointers
        }
        printf("\n");
    }
}

} // namespace nm

#endif
