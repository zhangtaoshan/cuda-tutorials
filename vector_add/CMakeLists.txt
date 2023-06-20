project(vector_add CUDA)

add_executable(vector_add vector_add.cu)

set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
