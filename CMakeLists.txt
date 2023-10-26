cmake_minimum_required(VERSION 3.26)
project(NumMKL)

set(CMAKE_CXX_STANDARD 20)
set(MKL_ROOT "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
set(TBB_ROOT "C:/Program Files (x86)/Intel/oneAPI/tbb/latest/lib/intel64/vc14")
set(COMPILER_ROOT "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64")
include_directories(${MKL_ROOT}/include)
link_directories(${MKL_ROOT}/lib/intel64)
link_directories(${COMPILER_ROOT})

set(omp 0)
#link_directories(${TBB_ROOT})
#link_libraries(mkl_intel_ilp64.lib mkl_tbb_thread.lib mkl_core.lib tbb12.lib)
if (omp)
    link_libraries(mkl_intel_ilp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib)
else ()
    link_libraries(mkl_intel_ilp64.lib mkl_sequential.lib mkl_core.lib)
endif ()

add_compile_options(/DMKL_ILP64)

if (MSVC)
    #add_compile_options(/O1)
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