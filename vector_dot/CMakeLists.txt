project(vector_dot CUDA)

add_executable(vector_dot_demo vec_dot.cu)

set_target_properties(vector_dot_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

