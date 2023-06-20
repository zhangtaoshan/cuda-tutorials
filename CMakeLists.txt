cmake_minimum_required(VERSION 3.10)

project(cuda-tutorials)

set(CMAKE_CUDA_ARCHITECTURES 61)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(vector_add)
add_subdirectory(vector_dot)
add_subdirectory(matrix_multiplication)
add_subdirectory(matrix_transpose)
