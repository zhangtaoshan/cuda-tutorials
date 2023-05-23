# 向量的逐元素相加
## vector_add_serial
在 CPU 上执行，逐个元素通过循环计算。

## vector_add_gpu_1
在 GPU 上执行，使用 1 个 block 和 1 个线程，实现过程类似于在 CPU 上串行执行。

## vector_add_gpu_2
在 GPU 上执行，使用 1 个 block 和多个线程。

## vector_add_gpu_3
在 GPU 上执行，使用多个 block 和多个线程。
```cu
// 当前线程在全局网格内的索引
int tid = blockIdx.x * blockDim.x + threadIdx.x;
// 如果一次给定的线程没有指定完给定数据，下次跳转的步长
int t_n = gridDim.x * blockDim.x
```

