## matrix_multiplication_cpu_1
在普通卷积中，输入矩阵 A 的 1 行与输入矩阵 B 的 1 列相乘，此时矩阵 B 的存取不连续从而影响 cache 命中。这里采取交换内部两层循环的方式，使得取 B 矩阵的元素变得连续。

## matrix_multiplication_cpu_2
同上，为了避免取 B 矩阵元素不连续的问题，现将 B 矩阵转置。这样，两个矩阵乘法的过程变成矩阵 A 的 1 行乘以矩阵 B 的 1 行，存取元素都变得连续。

## matrix_multiplication_gpu_1
使用足够多的线程块，沿变化的维度一次性将矩阵乘法计算完毕。

## matrix_multiplication_gpu_2 
每个线程块计算得到结果矩阵的 1 行，即计算输入矩阵 A 的 1 行和输入矩阵 B 的 1 列。   

## matrix_multiplication_gpu_3
使用共享内存，将矩阵 A 的 1 行放入共享内存中，其他的类似。

## matrix_multiplication_gpu_4
每次计算矩阵 C 的某 1 块，而非按照行或列计算，计算形式类似于 1 个棋盘状。

## matrix_multiplication_gpu_5
除了移除判断外，其他的同上面算法一致。
