//
// Created by Semigroup on 2023/10/19.
//

#ifndef NUMMKL_MARTIX_OPERATORS_HPP
#define NUMMKL_MARTIX_OPERATORS_HPP
#include "matrix_core.hpp"
#include "mkl_cblas.h"

namespace nm {

	template<typename T>
	Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		if (lhs.shape != rhs.shape) {
			// Return an empty matrix (or handle size mismatch error)
			return Matrix<T>(0,0);
		}

		Matrix<T> result(lhs.shape[0], lhs.shape[1]);
		for (int i = 0; i < lhs.shape[0] * lhs.shape[1]; ++i) {
			result.data[i] = lhs.data[i] + rhs.data[i];
		}
		return result;
	}

	template<typename T>
	Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
		if (lhs.shape != rhs.shape) {
			// Return an empty matrix (or handle size mismatch error)
			return Matrix<T>(0,0);
		}

		Matrix<T> result(lhs.shape[0], lhs.shape[1]);
		for (int i = 0; i < lhs.shape[0] * lhs.shape[1]; ++i) {
			result.data[i] = lhs.data[i] - rhs.data[i];
		}
		return result;
	}

	// For double precision
	Matrix<double> operator*(const Matrix<double>& lhs, const Matrix<double>& rhs) {
		if (lhs.shape[1] != rhs.shape[0]) {
			// Return an empty matrix (or handle size mismatch error)
			return Matrix<double>(0,0);
		}

		Matrix<double> result(lhs.shape[0], rhs.shape[1]);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					lhs.shape[0], rhs.shape[1], lhs.shape[1],
					1.0, lhs.data, lhs.shape[1],
					rhs.data, rhs.shape[1],
					0.0, result.data, result.shape[1]);
		return result;
	}

	// For single precision
	Matrix<float> operator*(const Matrix<float>& lhs, const Matrix<float>& rhs) {
		if (lhs.shape[1] != rhs.shape[0]) {
			// Return an empty matrix (or handle size mismatch error)
			return Matrix<float>(0,0);
		}

		Matrix<float> result(lhs.shape[0], rhs.shape[1]);

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					lhs.shape[0], rhs.shape[1], lhs.shape[1],
					1.0f, lhs.data, lhs.shape[1],
					rhs.data, rhs.shape[1],
					0.0f, result.data, result.shape[1]);
		return result;
	}

	Matrix<int> operator*(const Matrix<int>& lhs, const Matrix<int>& rhs) {
		if (lhs.shape[1] != rhs.shape[0]) {
			// Return an empty matrix (or handle size mismatch error)
			return Matrix<int>(0,0);
		}

		// Create temporary float matrices for casting
		Matrix<float> lhsFloat(lhs.shape[0], lhs.shape[1]);
		Matrix<float> rhsFloat(rhs.shape[0], rhs.shape[1]);

		// Cast int data to float
		for (int i = 0; i < lhs.shape[0] * lhs.shape[1]; ++i) {
			lhsFloat.data[i] = static_cast<float>(lhs.data[i]);
		}
		for (int i = 0; i < rhs.shape[0] * rhs.shape[1]; ++i) {
			rhsFloat.data[i] = static_cast<float>(rhs.data[i]);
		}

		// Multiply using MKL
		Matrix<float> resultFloat(lhs.shape[0], rhs.shape[1]);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					lhs.shape[0], rhs.shape[1], lhs.shape[1],
					1.0f, lhsFloat.data, lhs.shape[1],
					rhsFloat.data, rhs.shape[1],
					0.0f, resultFloat.data, resultFloat.shape[1]);

		// Cast float result back to int
		Matrix<int> resultInt(lhs.shape[0], rhs.shape[1]);
		for (int i = 0; i < resultInt.shape[0] * resultInt.shape[1]; ++i) {
			resultInt.data[i] = static_cast<int>(resultFloat.data[i]);
		}

		return resultInt;
	}
}  // namespace nm

#endif
