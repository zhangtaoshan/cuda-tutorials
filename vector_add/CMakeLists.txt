project(vector_add CUDA)

add_executable(vector_add_demo vec_add.cu)

set_target_properties(vector_add_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

