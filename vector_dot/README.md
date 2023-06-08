# 两个向量间的点乘

## vector_dot_serial
在 CPU 上执行，逐个元素相乘并相加。

## vector_dot_product_gpu_1
在 GPU 上执行，使用 1 个 block 和多个线程，使用分散规约求向量内积。共享内存大小与单个 block 内的线程数相同，每个线程有自己的变量 `temp`。由此，在数据总数大于线程数时，不会出现 `temp` 被覆盖的情况。

```c
for (int tid = tidx; tid < NUM_INPUT; tid += t_n)
{
    temp += a[tid] * b[tid];
}
```
然后使用 `__syncthreads()` 同步，等待所有线程执行完成。在分散规约中，每次两相邻元素进行求和。这里，使用变量 `i` 确定偶数元素，即前一个元素；使用变量 `j` 确定后一个元素。以上操作均在同一个 `block` 内完成。这里，共享内存的大小为 `THREADS`，而数据大小为 `NUM_INPUT`，由于xxxx，所以不会发生共享内存区域的覆盖。
```c
int i = 2, j = 1;
while (i <= THREADS)
{
    if ((tidx % i) == 0)
    {
        tmp[tidx] += tmp[tidx + j];
    }
    __syncthreads();
    i *= 2;
    j *= 2;
}
```
最后，取共享内存的第一个位置元素即为最终结果。
## vector_dot_product_gpu_2
在 GPU 上运行，使用低线程规约的方式求和，其他设置同上。在共享内存的数组中，左端部分的值和来自右端部分的值一一相加。
```c
int i = NUM_INPUT / 2;
while (i != 0)
{
    // left elements added from right elements
    if (tidx < i)
    {
        tmp[tidx] += tmp[tidx + i];
    }
    __syncthreads();
    i /= 2;
}
```
## vector_dot_product_gpu_3
