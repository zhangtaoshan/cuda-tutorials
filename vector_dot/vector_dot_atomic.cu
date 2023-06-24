#include "../common/utils.h"

// 输入元素数量
#define NUM_INPUT 2048
// block的数量
#define BLOCKS 2
// 单个block中的线程数量
#define THREADS 128


// 使用原子操作
__global__ void vector_dot_5(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    // 初始化为零
    if ((threadIdx.x == 0) && (blockIdx.x == 0))
    {
	    c[0] = 0.0;
    }
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    for (; tid < NUM_INPUT; tid += t_n)
    {
	    temp += a[tid] * b[tid];
    }
    // 使用原子操作将各temp归约到c中，此时各个block内的元素还没有归约好
    atomicAdd(c, temp);
}


__global__ void vector_dot_6(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    // 初始化为零
    if ((threadIdx.x == 0) && (blockIdx.x == 0))
    {
	    c[0] = 0.0;
    }
    __shared__ DATATYPE tmp[THREADS];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    for (; tid < NUM_INPUT; tid += t_n)
    {
	    temp += a[tid] * b[tid];
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (tidx < i)
        {
            tmp[tidx] += tmp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    // 使用原子操作将各temp归约到c中
    if (tidx == 0)
    {
	    atomicAdd(c, tmp[0]);
    }
}


int main()
{
    // 输入元素所占空间
    size_t size = sizeof(DATATYPE) * NUM_INPUT;
    // 为输入输出分配内存并初始化
    DATATYPE* h_a = (DATATYPE*)malloc(size);
    DATATYPE* h_b = (DATATYPE*)malloc(size);
    for (int i = 0; i < NUM_INPUT; ++i)
    {
        h_a[i] = rand() / (DATATYPE)RAND_MAX;
        h_b[i] = rand() / (DATATYPE)RAND_MAX;
    }
    DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE));
    // baseline
    DATATYPE* baseline = (DATATYPE*)malloc(sizeof(DATATYPE));
    vector_dot_baseline(h_a, h_b, baseline, NUM_INPUT);
    // 分配设备上的内存并拷贝输入数据
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size);
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    DATATYPE* d_c = NULL;
    cudaMalloc((void**)&d_c, sizeof(DATATYPE));
    cudaError_t err = cudaGetLastError();
    if (err != 0) {
        printf("Error in cudaMalloc and cudaMemcpy: %s.\n", 
            cudaGetErrorString(err));
    }
    // 定义启动核函数的参数
    dim3 blocksPerGrid(BLOCKS, 1, 1);
    dim3 threadsPerBlock(THREADS, 1, 1);
    {
        vector_dot_5<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c);
        err = cudaGetLastError();
        if (err != 0) {
            printf("Error in forward: %s.\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
        check_value(baseline, h_c);
    }
    {
        vector_dot_6<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c);
        err = cudaGetLastError();
        if (err != 0) {
            printf("Error in forward: %s.\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
        check_value(baseline, h_c);
    }
    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    err = cudaGetLastError();
    if (err != 0) {
        printf("Error in cudaFree: %s.\n", cudaGetErrorString(err));
    }
}
