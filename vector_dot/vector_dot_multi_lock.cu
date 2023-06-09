#include "../common/utils.h"

// 输入元素数量
#define NUM_INPUT 2048
// block的数量
#define BLOCKS 2
// 单个block中的线程数量
#define THREADS 128


// 使用多个block，最后规约在CPU上做，每个block内部发生了和gpu_2中相同的操作
__global__ void vector_dot_product_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c_temp)
{
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
    int i = NUM_INPUT / 2;
    while (i != 0)
    {
        if (tidx < i)
        {
            tmp[tidx] += tmp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    // 将共享内存的第一个数值赋值到结果数组中
    if (tidx == 0)
    {
	    c_temp[bidx] = tmp[0];
    }
}


// 使用多个block，最后规约在GPU上做
__global__ void vector_dot_product_gpu_4(DATATYPE* c_temp, DATATYPE* c)
{
    // 共享内存大小声明为block的数量
    __shared__ DATATYPE tmp[BLOCKS];
    const int tidx = threadIdx.x;
    // 每个block内仅使用一个线程做规约
    tmp[tidx] = c_temp[tidx];
    __syncthreads();
    // 和上面的低线程规约一样
    int i = BLOCKS / 2;
    while (i != 0)
    {
        if (tidx < i)
        {
            tmp[tidx] += tmp[tidx + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (tidx == 0)
    {
	    c[0] = tmp[0];
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
    // 在CPU上规约
    { 
        vector_dot_product_gpu_3<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c);
        err = cudaGetLastError();
        if (err != 0) {
            printf("Error in forward: %s.\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
        DATATYPE temp = 0.0;
        for (int i = 0; i < BLOCKS; ++i)
        {
            temp += h_c[i];
        }
        printf("result: %f\n", temp);
    }
    // 在GPU上规约
    {
        vector_dot_product_gpu_3<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c);
        err = cudaGetLastError();
        if (err != 0) {
            printf("Error in forward: %s.\n", cudaGetErrorString(err));
        }
        DATATYPE* dd_c = NULL;
        cudaMalloc((void**)&dd_c, sizeof(DATATYPE));
        vector_dot_product_gpu_4<<<1, blocksPerGrid>>>(d_c, dd_c);
        // memory copy
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
        printf("result: %f\n", *h_c);
        cudaFree(dd_c);
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

