#pragma once
#include "mkl.h"
#include <array>
#include <initializer_list>

namespace nm {

	template<typename T>
	class Matrix {
	private:
		std::array<int, 2> _shape{};
		bool gc = true;

		struct RowOperator {
			T* row_data;
			RowOperator(T* data) : row_data(data) {}
			T& operator[](int j) {
				return row_data[j];
			}
			const T& operator[](int j) const {
				return row_data[j];
			}
		};

	public:
		T* data;
		const std::array<int, 2>& shape = _shape;

		Matrix(int i, int j) : _shape({i,j})
		{
			data = (T*)mkl_malloc(i * j * sizeof(T), 64);
		}

		Matrix(int i, int j, void* data) : _shape({i,j}), data((T*)data)
		{
			gc = false;
		}

		// Constructor from initializer list
		Matrix(std::initializer_list<std::initializer_list<T>> list) {
			int rows = list.size();
			int cols = list.begin()->size();
			_shape = {rows, cols};
			//data = new T[rows * cols];
			data = (T*)mkl_malloc(rows * cols * sizeof(T), 64);
			int idx = 0;
			for (const auto& row : list) {
				for (const auto& val : row) {
					data[idx++] = val;
				}
			}
		}

		// Copy constructor
		Matrix(const Matrix& other) : _shape(other._shape) {
			data = mkl_malloc(other.shape[0] * other.shape[1] * sizeof(T), 64);
			const size_t size = other.shape[0] * other.shape[1];
			for (int i = 0; i < size; ++i) {
				data[i] = other.data[i];
			}
		}

		// Move constructor
		Matrix(Matrix&& other) noexcept : _shape(other._shape) , data(other.data) {
			other.data = nullptr;
			other._shape = {0,0};
		}

		// Copy assignment operator
		Matrix& operator=(const Matrix& other) {
			if (this != &other) {
				if (gc) mkl_free(data);
				_shape = other._shape;
				data = mkl_malloc(other.shape[0] * other.shape[1] * sizeof(T), 64);
				for (int i = 0; i < other.shape[0] * other.shape[1]; ++i) {
					data[i] = other.data[i];
				}
			}
			return *this;
		}

		// Move assignment operator
		Matrix& operator=(Matrix&& other) noexcept {
			if (this != &other) {
				if (gc) mkl_free(data);
				_shape = other._shape;
				data = other.data;

				other.data = nullptr;
				other._shape = {0,0};
			}
			return *this;
		}

		RowOperator operator[](int i) {
			return RowOperator(data + i * _shape[1]);
		}

		RowOperator operator[](int i) const {
			return RowOperator(data + i * _shape[1]);
		}

		T& operator()(int i, int j) {
			return data[i * _shape[1] + j];
		}

		const T& operator()(int i, int j) const {
			return data[i * _shape[1] + j];
		}

		~Matrix() {
			if (gc) mkl_free(data);
		}
	};

}  // namespace nm
