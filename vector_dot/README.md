# 两个向量间的点乘

## vector_dot_serial
在 CPU 上执行，逐个元素相乘并相加。

## vector_dot_product_gpu_1
在 GPU 上执行，使用 1 个 block 和多个线程，使用分散规约求向量内积。
