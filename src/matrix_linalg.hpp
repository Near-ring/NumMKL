//
// Created by Semigroup on 2023/10/23.
//

#ifndef NUMMKL_MATRIX_LINALG_HPP
#define NUMMKL_MATRIX_LINALG_HPP

#include "matrix_core.hpp"
#include "mkl.h"

namespace nm::linalg {
inline void lu(Matrix<float>& A)
{
    LAPACKE_mkl_sgetrfnp(LAPACK_ROW_MAJOR, A.shape[0], A.shape[1], A.data, A.shape[1]);
}

float sse(Matrix<float>& A, Matrix<float>& B)
{
    float result = 0.0;
    for (int i = 0; i < A.shape[0] * A.shape[1]; ++i) {
        auto e = A.data[i] - B.data[i];
        result += e * e;
    }
    return result;
}
} // namespace nm::linalg

#endif // NUMMKL_MATRIX_LINALG_HPP
