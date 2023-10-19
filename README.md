
## `NumMKL` - A C++ Wrapper for Intel MKL

---

### Introduction

`NumMKL` is a C++ wrapper around Intel's Math Kernel Library (MKL) aimed at providing an easy-to-use matrix manipulation library reminiscent of Python's NumPy. Built with performance in mind, `NumMKL` leverages the power of Intel MKL to deliver fast and efficient matrix operations.

---

### Current Features (in development)
- **NumPy (Python)**:

  ```python
  import numpy as np
  A = np.empty((3, 3), dtype=np.float64)
  B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
  ```
- **NumMKL**: Create matrices using standard constructors or initializer lists.
  
  ```cpp
  nm::Matrix<double> A(3, 3);
  nm::Matrix<double> B = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
  ```

- Access matrix elements using intuitive indexing.

  ```cpp
  double value = B[1][2];
  double value = B(1, 2);  // Accesses the element in the second row and third column.
  ```

- Perform basic operations such as addition and multiplication with ease.

  ```cpp
  auto C = A + B;
  auto D = A * B;
  ```

### Getting Started / Installation

#### Pre-requisites:
1. **Intel MKL**: Before you can use `NumMKL`, you need to install Intel's Math Kernel Library (MKL).

  - **Installing Intel HPC Toolkit**:
    Visit [Intel HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) to download and install the toolkit, which includes MKL.

  - **MKL Link Line Advisor**:
    For advanced users and to obtain specific link lines based on your configuration, use Intel's [MKL Link Line Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html).


