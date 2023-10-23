//
// Created by Semigroup on 2023/10/23.
//

#ifndef NUMMKL_MATRIX_LINALG_HPP
#define NUMMKL_MATRIX_LINALG_HPP

#include "matrix_core.hpp"
#include "mkl.h"

namespace nm::linalg {
inline void lu(Matrix<float> const& mat)
{
    LAPACKE_mkl_sgetrfnp(LAPACK_ROW_MAJOR, mat.shape[0], mat.shape[1], mat.data, mat.shape[1]);
}

inline float norm(Matrix<float> const& mat, char norm_type)
{
    return LAPACKE_slange(LAPACK_ROW_MAJOR, norm_type, mat.shape[0], mat.shape[1], mat.data,
                          mat.shape[1]);
}


} // namespace nm::linalg

#endif // NUMMKL_MATRIX_LINALG_HPP
