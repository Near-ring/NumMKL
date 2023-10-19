//
// Created by Semigroup on 2023/10/19.
//

#ifndef NUMCPP_MATRIX_RANDOM_HPP
#define NUMCPP_MATRIX_RANDOM_HPP

#include "matrix_core.hpp"
#include <mkl_vsl.h>

namespace nm {

	class RandomGenerator {
	private:
		VSLStreamStatePtr stream = nullptr;

	public:
		RandomGenerator() {
			vslNewStream(&stream, VSL_BRNG_MT19937, 777);
		}

		~RandomGenerator() {
			vslDeleteStream(&stream);
		}

		void fillUniformRandom(Matrix<double>& matrix) {
			vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, matrix.shape[0] * matrix.shape[1], matrix.data, 0.0, 1.0);
		}
		void fillUniformRandom(Matrix<float>& matrix) {
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, matrix.shape[0] * matrix.shape[1], matrix.data, 0.0, 1.0);
		}
	};

	template<typename T>
	Matrix<T> randMatrix(int i, int j) {
		Matrix<T> result(i, j);
		RandomGenerator gen;
		gen.fillUniformRandom(result);
		return result;
	}

} // namespace nm
#endif
