cmake_minimum_required(VERSION 3.26)
project(NumMKL)

set(CMAKE_CXX_STANDARD 20)
set(MKL_ROOT "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
set(MKL_THREADING_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64")
include_directories(${MKL_ROOT}/include)
link_directories(${MKL_ROOT}/lib/intel64)
link_directories(${MKL_THREADING_ROOT})

set(OMP 1)
if (OMP)
    link_libraries(mkl_intel_ilp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib)
else ()
    link_libraries(mkl_intel_ilp64.lib mkl_sequential.lib mkl_core.lib)
endif ()

add_compile_options(/DMKL_ILP64)

if (MSVC AND CMAKE_BUILD_TYPE MATCHES "Release")
    add_compile_options(/O1)
endif ()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()

add_executable(
        NumMKL
        src/matrix_core.hpp
        src/matrix_core.hpp
        src/matrix_operators.hpp
        src/matrix_random.hpp
        src/matrix_random.hpp
        src/basic_test.cpp
        src/avx_utils.hpp
        src/lu.hpp
        src/matrix_linalg.hpp
        src/matrix_utils.hpp
        src/cpp_utils.hpp
        src/num_mkl.hpp
)