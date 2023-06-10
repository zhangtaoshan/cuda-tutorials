#include "../common/utils.h"

// 输入元素数量
#define NUM_INPUT 2048
// block的数量
#define BLOCKS 2
// 单个block中的线程数量
#define THREADS 128
// 
__device__ unsigned int lockcount = 0;


__device__ void vector_dot(DATATYPE* out, DATATYPE* temp)
{
    // 归约一个block内的所有线程值，并存放到out中
    const int tidx = threadIdx.x;
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (tidx < i)
        {
            temp[tidx] += temp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (tidx == 0)
    {
	    out[0] = temp[0];
    }
}


__global__ void vector_dot_7(DATATYPE* a, DATATYPE* b, DATATYPE* c, DATATYPE* c_temp)
{
    // 常规操作将各元素存入线程自己的temp中后存入共享内存中
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
    // 规约各个block内的线程值
    vector_dot(&c_temp[blockIdx.x], tmp);
    __shared__ bool lock;
    __threadfence();
    if (tidx == 0)
    {
        unsigned int lockiii = atomicAdd(&lockcount, 1);
        lock = (lockcount == gridDim.x);
    }
    __syncthreads();
    if (lock)
    {
        tmp[tidx] = c_temp[tidx];
        __syncthreads();
        vector_dot(c, tmp);
        lockcount = 0;
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
    // 使用多个block，每个block单独计算自己的内容
    DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE) * BLOCKS);
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
    cudaMalloc((void**)&d_c, sizeof(DATATYPE) * BLOCKS);
    cudaError_t err = cudaGetLastError();
    if (err != 0) {
        printf("Error in cudaMalloc and cudaMemcpy: %s.\n", 
            cudaGetErrorString(err));
    }
    // 定义启动核函数的参数
    dim3 blocksPerGrid(BLOCKS, 1, 1);
    dim3 threadsPerBlock(THREADS, 1, 1);
    DATATYPE* dd_c = NULL;
    cudaMalloc((void**)&dd_c, sizeof(DATATYPE) * BLOCKS);
    vector_dot_7<<<blocksPerGrid, threadsPerBlock>>>(
        d_a, d_b, d_c, dd_c);
    err = cudaGetLastError();
    if (err != 0) {
        printf("Error in forward: %s.\n", cudaGetErrorString(err));
    }
    // memory copy
    cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
    check_value(baseline, h_c);
    cudaFree(dd_c);
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
