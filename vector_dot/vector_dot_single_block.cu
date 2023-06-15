#include "../common/utils.h"

// 输入元素数量
#define NUM_INPUT 2048
// 单个block中的线程数量
#define THREADS 128


// 使用单个block多个线程，使用分散归约
__global__ void vector_dot_1(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    // 共享内存大小定义为线程数
    __shared__ DATATYPE tmp[THREADS];
    const int tidx = threadIdx.x;
    // 如果当前线程数少于数据，则需要线程作第多次计算，定义步长
    const int t_n = blockDim.x;
    // 每个线程有自己的变量
    double temp = 0.0;
    // 每次处理线程数个元素，如索引为0的线程的temp累加了0, 0+THREADS, 0+2*THREADS...的元素
    for (int tid = tidx; tid < NUM_INPUT; tid += t_n)
    {
	    temp += a[tid] * b[tid];
    }
    // 为共享空间的每个位置赋值
    tmp[tidx] = temp;
    __syncthreads();
    // 对THREADS个元素归约
    int i = 2, j = 1;
    // 进行log(THREADS)次计算
    while (i <= THREADS)
    {
        if ((tidx % i) == 0)
	    {
            // 后一个归约到当前
	        tmp[tidx] += tmp[tidx + j];
	    }
        __syncthreads();
        i *= 2;
        j *= 2;
    }
    // 归约完成后第一个位置的值即为最终答案
    if (tidx == 0)
    {
	    c[0] = tmp[0];
    }
}


// 使用单个block多个线程，使用低线程归约
__global__ void vector_dot_2(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    __shared__ DATATYPE tmp[THREADS];
    int tidx = threadIdx.x;
    const int t_n = blockDim.x;
    double temp = 0.0;
    for (int tid = tidx; tid < NUM_INPUT; tid += t_n)
    {
	    temp += a[tid] * b[tid];
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = THREADS / 2;
    while (i != 0)
    {
        // 把线程分为小于i和大于i的两部分
        if (tidx < i)
        {
            // 后面部分的元素以相同步长归约到前面元素
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
    dim3 blocksPerGrid(1, 1, 1);
    dim3 threadsPerBlock(THREADS, 1, 1);
    // 使用分散归约
    {
        vector_dot_1<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c);
        err = cudaGetLastError();
        if (err != 0) {
            printf("Error in forward: %s.\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
        check_value(baseline, h_c);
    }
    // 使用低线程归约
    {
        vector_dot_2<<<blocksPerGrid, threadsPerBlock>>>(
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
