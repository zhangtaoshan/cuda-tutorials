#include "../common/utils.h"

// 输入元素数量
#define NUM_INPUT 2048
// block的数量
#define BLOCKS 2
// 单个block中的线程数量
#define THREADS 128


__global__ void vector_add_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n)
{
    int tid = threadIdx.x;
    const int t_n = blockDim.x;
    for (; tid < n; tid += t_n)
    {
        c[tid] = a[tid] + b[tid];
    }
}


__global__ void vector_add_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int t_n = gridDim.x * blockDim.x;
    for (; tid < n; tid += t_n)
    {
        c[tid] = a[tid] + b[tid];
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
    DATATYPE* h_c = (DATATYPE*)malloc(size);
    // baseline
    DATATYPE* baseline = (DATATYPE*)malloc(size);
    vector_add_baseline(h_a, h_b, baseline, NUM_INPUT);
    // 分配设备上的内存并拷贝输入数据
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size);
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    DATATYPE* d_c = NULL;
    cudaMalloc((void**)&d_c, size);
    cudaError_t err = cudaGetLastError();
    if (err != 0) {
        printf("Error in cudaMalloc and cudaMemcpy: %s.\n", 
            cudaGetErrorString(err));
    }
    // 使用单block多线程
    {
        // 定义启动核函数的参数
        dim3 blocksPerGrid(1, 1, 1);
        dim3 threadsPerBlock(THREADS, 1, 1);
        vector_add_1<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c, NUM_INPUT);
        err = cudaGetLastError();
        if (err != 0) {
            printf("Error in forward: %s.\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE) * NUM_INPUT, cudaMemcpyDeviceToHost);
        check_vector(baseline, h_c, NUM_INPUT);
    }
    // 使用多个block
    {
        // 定义启动核函数的参数
        dim3 blocksPerGrid(BLOCKS, 1, 1);
        dim3 threadsPerBlock(THREADS, 1, 1);
        vector_add_2<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c, NUM_INPUT);
        err = cudaGetLastError();
        if (err != 0) {
            printf("Error in forward: %s.\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE) * NUM_INPUT, cudaMemcpyDeviceToHost);
        check_vector(baseline, h_c, NUM_INPUT);
    }
    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    free(baseline);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    err = cudaGetLastError();
    if (err != 0) {
        printf("Error in cudaFree: %s.\n", cudaGetErrorString(err));
    }
}
