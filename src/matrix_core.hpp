#pragma once
#include "mkl.h"
#include <array>
#include <cstdio>
#include <initializer_list>

namespace nm {

typedef char i8;
typedef short i16;
typedef int i32;
typedef long long int i64;
typedef unsigned char byte;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long int u64;
typedef float f32;
typedef double f64;

template <typename Type>
class Matrix
{
  private:
    Type* _data = nullptr;
    bool gc = true;
    int _lda = 0;
    std::array<int, 2> _shape{};

  public:
    Type* const& data = _data;
    const int& lda = _lda;
    const std::array<int, 2>& shape = _shape;

    Matrix(int i, int j) : _shape({i, j}), _lda(j)
    {
        _data = (Type*)mkl_malloc(i * j * sizeof(Type), 64);
    }

    Matrix(int i, int j, Type* data, bool gc = false) : _shape({i, j}), _data(data), _lda(j), gc(gc)
    {
    }

    Matrix(int i, int j, int lda, Type* data, bool gc = false)
        : _shape({i, j}), _data(data), _lda(lda), gc(gc)
    {
    }

  private:
    inline void deep_copy(const Matrix& other) {
        for (int i = 0; i < _shape[0]; ++i) {
            for (int j = 0; j < _shape[1]; ++j) {
                _data[i * _lda + j] = other._data[i * other._lda + j];
            }
        }
    }

  public:
    // Constructor from initializer list
    Matrix(std::initializer_list<std::initializer_list<Type>> list)
    {
        int rows = list.size();
        int cols = list.begin()->size();
        _shape = {rows, cols};
        _lda = cols;
        _data = static_cast<Type*>(mkl_malloc(rows * cols * sizeof(Type), 64));
        //_data = (Type*)_aligned_malloc(rows * cols * sizeof(Type), 256);
        printf("data: %p\n", _data);
        int idx = 0;
        for (const auto& row : list) {
            for (const auto& val : row) {
                _data[idx++] = val;
            }
        }
    }

    // Move constructor
    Matrix(Matrix&& other) noexcept
        : _shape(other._shape), _data(other._data), _lda(other._lda), gc(other.gc)
    {
        other._lda = 0;
        other._data = nullptr;
        other._shape = {0, 0};
    }

    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept
    {
        if (this != &other) {
            if (gc) mkl_free(_data);
            _shape = other._shape;
            _data = other._data;
            _lda = other._lda;
            gc = other.gc;

            other._data = nullptr;
            other._shape = {0, 0};
            other._lda = 0;
        }
        return *this;
    }

    // Copy constructor
    Matrix(const Matrix& other) : _shape(other._shape), gc(true)
    {
        const size_t size = _shape[0] * _shape[1];
        _data = (Type*)mkl_malloc(size * sizeof(Type), 64);

        if (other._shape[1] != other._lda) {
            _lda = other._shape[1];
            deep_copy(other);
        }
        else {
            _lda = other._lda;
            std::memcpy(_data, other._data, size * sizeof(Type));
        }
    }

    // Copy assignment
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            if (_shape == other._shape) {
                deep_copy(other);
                return *this;
            }
            const size_t size = _shape[0] * _shape[1];
            const size_t size_other = other._shape[0] * other._shape[1];
            if (size != size_other)
            {
                if (gc) mkl_free(_data);
                _data = (Type*)mkl_malloc(size_other * sizeof(Type), 64);
                gc = true;
            }
            _shape = other._shape;
            _lda = other._lda;
            deep_copy(other);
        }
        return *this;
    }

    inline Type& operator()(int i, int j) { return data[i * _lda + j]; }
    inline Type& operator()(int i, int j) const { return data[i * _lda + j]; }

    inline Type* operator[](int i) { return _data + i * _lda; }
    inline Type* operator[](int i) const { return _data + i * _lda; }

    bool operator==(const Matrix<Type>& other) const
    {
        if (_shape != other._shape) return false;
        for (int i = 0; i < _shape[0]; ++i) {
            for (int j = 0; j < _shape[1]; ++j) {
                if (*(_data + i * _lda + j) != *(other._data + i * other._lda + j)) return false;
            }
        }
        return true;
    }

    void reshape(int i, int j) {}

    Matrix<Type> operator+(const Matrix<Type>& B) const;
    Matrix<Type> operator-(const Matrix<Type>& B) const;
    Matrix<float> operator*(const Matrix<float>& B) const;
    Matrix<double> operator*(const Matrix<double>& B) const;
    [[nodiscard]] float min() const;
    [[nodiscard]] float max() const;

    ~Matrix()
    {
        if (gc) mkl_free(_data);
    }
};
} // namespace nm
