#pragma once
#include "mkl.h"
#include <array>
#include <initializer_list>
#include <iomanip>
#include <sstream>

namespace nm
{

	template<typename Type>
	class Matrix
	{
	private:
		std::array<int, 2> _shape{};
		bool gc = true;
		Type* _data;

	public:
		Type*& data = _data;
		const std::array<int, 2>& shape = _shape;

		Matrix(int i, int j)
			: _shape({i, j})
		{
			_data = (Type*)mkl_malloc(i * j * sizeof(Type), 64);
		}

		Matrix(int i, int j, void* data, bool gc = false)
			: _shape({i, j}), _data((Type*)data)
		{
			this->gc = gc;
		}

		// Constructor from initializer list
		Matrix(std::initializer_list<std::initializer_list<Type>> list)
		{
			int rows = list.size();
			int cols = list.begin()->size();
			_shape = {rows, cols};
			//_data = new Type[rows * cols];
			_data = (Type*)mkl_malloc(rows * cols * sizeof(Type), 64);
			int idx = 0;
			for (const auto& row: list)
			{
				for (const auto& val: row)
				{
					_data[idx++] = val;
				}
			}
		}

		// Copy constructor
		Matrix(const Matrix& other)
			: _shape(other._shape)
		{
			_data = (Type*)mkl_malloc(other.shape[0] * other.shape[1] * sizeof(Type), 64);
			const size_t size = other.shape[0] * other.shape[1];
			for (int i = 0; i < size; ++i)
			{
				_data[i] = other._data[i];
			}
		}

		// Move constructor
		Matrix(Matrix&& other) noexcept
			: _shape(other._shape), _data(other._data)
		{
			other._data = nullptr;
			other._shape = {0, 0};
		}

		// Copy assignment operator
		Matrix& operator=(const Matrix& other)
		{
			if (this != &other)
			{
				if (gc) mkl_free(_data);
				_shape = other._shape;
				_data = (Type*)mkl_malloc(other.shape[0] * other.shape[1] * sizeof(Type), 64);
				for (int i = 0; i < other.shape[0] * other.shape[1]; ++i)
				{
					_data[i] = other._data[i];
				}
			}
			return *this;
		}

		// Move assignment operator
		Matrix& operator=(Matrix&& other) noexcept
		{
			if (this != &other)
			{
				if (gc) mkl_free(_data);
				_shape = other._shape;
				_data = other._data;

				other._data = nullptr;
				other._shape = {0, 0};
			}
			return *this;
		}

		inline Type* operator[](int i)
		{
			return _data + i * _shape[1];
		}
		inline Type* operator[](int i) const
		{
			return _data + i * _shape[1];
		}

		inline const Type& operator()(int i)
		{
			return _data[i];
		}
		inline const Type& operator()(int i) const
		{
			return _data[i];
		}

		inline const Type& operator()(int i, int j)
		{
			return _data[i * _shape[1] + j];
		}
		inline const Type& operator()(int i, int j) const
		{
			return _data[i * _shape[1] + j];
		}

		[[nodiscard]] Matrix<Type> operator()(const std::array<int, 2>& row_range, const std::array<int, 2>& col_range) const
		{
			if (row_range[0] > row_range[1] || col_range[0] > col_range[1] || row_range[0] < 0 || col_range[0] < 0 || row_range[1] >= _shape[0] || col_range[1] >= _shape[1])
			{
				//throw std::invalid_argument("Invalid range");
				printf("Invalid range\n");
			}

			int rows = row_range[1] - row_range[0] + 1;// Adjusted for inclusive end
			int cols = col_range[1] - col_range[0] + 1;// Adjusted for inclusive end
			Matrix<Type> result(rows, cols);

			if (_shape[1] < 16)
			{
				auto rd = result._data;
				for (int i = 0; i < rows; ++i)
				{
					for (int j = 0; j < cols; ++j)
					{
						rd[i * cols + j] = _data[(i + row_range[0]) * _shape[1] + (j + col_range[0])];
					}
				}
			}
			else
			{
				for (int i = 0; i < rows; ++i)
				{
					Type* src_ptr = _data + (i + row_range[0]) * _shape[1] + col_range[0];
					Type* dest_ptr = result._data + i * cols;

					std::memcpy(dest_ptr, src_ptr, cols * sizeof(Type));
				}
			}

			return result;
		}

        bool operator==(const Matrix<Type>& other) const
        {
            if (_shape != other._shape) return false;
            int l = _shape[0] * _shape[1];
            for (int i = 0; i < l; ++i)
            {
                if (_data[i] != other._data[i]) return false;
            }
            return true;
        }

		void reshape(int i, int j)
		{
			if (i * j != _shape[0] * _shape[1])
			{
				//throw std::invalid_argument("Invalid shape");
				printf("WARNING: reshape - shape mismatch\n");
			}
			_shape = {i, j};
		}

		[[nodiscard]] Matrix<Type> transpose() const noexcept
		{
			Matrix<Type> result(_shape[1], _shape[0]);
			const size_t block = 32;
			for (size_t i = 0; i < _shape[0]; i += block)
			{
				for (size_t j = 0; j < _shape[1]; ++j)
				{
					for (size_t b = 0; b < block && i + b < _shape[0]; ++b)
					{
						result._data[j * _shape[0] + i + b] = _data[(i + b) * _shape[1] + j];
					}
				}
			}
			return result;
		}

		[[nodiscard]] std::string to_string() const
		{
			std::stringstream ss;

			// Find the maximum width of the entries.
			int max_width = 0;
			for (int i = 0; i < _shape[0]; ++i)
			{
				for (int j = 0; j < _shape[1]; ++j)
				{
					int current_width = std::to_string((*this)[i][j]).length();
					max_width = std::max(max_width, current_width);
				}
			}

			// Use the max width to format the output.
			for (int i = 0; i < _shape[0]; ++i)
			{
				for (int j = 0; j < _shape[1]; ++j)
				{
					ss << std::setw(max_width + 1) << (*this)[i][j];
				}
				ss << '\n';
			}

			return ss.str();
		}

		~Matrix()
		{
			if (gc) mkl_free(_data);
		}
	};
}// namespace nm
