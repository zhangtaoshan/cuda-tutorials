#include "../common/utils.h"

// 输入矩阵维度
#define INPUT_M 200
#define INPUT_N 700
#define INPUT_L 500
// 线程数和block数
#define THREADS 512
#define BLOCKS 4


__global__ void matrix_transposition_gpu_1d(DATATYPE* a, DATATYPE* b, int m, int n)
{ 
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    // 每个block处理矩阵的1行（列）
    while (bidx < m)
    {
        while (tidx < n)
        {
            b[tidx * m + bidx] = a[bidx * n + tidx];
            tidx += blockDim.x;
        }
        bidx += gridDim.x;
    }
}


int main()
{
    size_t size_a = sizeof(DATATYPE) * INPUT_M * INPUT_N;
    size_t size_b = sizeof(DATATYPE) * INPUT_N * INPUT_M;
    DATATYPE* h_a = (DATATYPE*)malloc(size_a);
    DATATYPE* h_b = (DATATYPE*)malloc(size_b);
    // 输入初始化
    for (int i = 0; i < INPUT_M; ++i)
    {
        for (int j = 0; j < INPUT_N; ++j)
        {
            h_a[i * INPUT_N + j] = rand() / (DATATYPE)RAND_MAX;
        }
    }
    // baseline
    DATATYPE* baseline = (DATATYPE*)malloc(size_b);
    matrix_transpose_baseline(h_a, baseline, INPUT_M, INPUT_N);
    // 分配设备上的内存
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size_a);
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size_b);
    cudaError_t err = cudaGetLastError();
    if (err != 0)
    {
        printf("error in cudaMalloc: %s\n", cudaGetErrorString(err));
    }
    // 数据拷贝
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != 0)
    {
        printf("error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    }
    // 使用grid内所有线程计算
    {
        // 定义启动核函数的参数
        dim3 blocksPerGrid(BLOCKS, 1, 1);
        dim3 threadsPerBlock(THREADS, 1, 1);
        matrix_transposition_gpu_1d<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, INPUT_M, INPUT_N);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("error in kernel forward: %s\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_b, d_b, size_b, cudaMemcpyDeviceToHost);
        check_matrix(baseline, h_b, INPUT_N, INPUT_M);
    }
    // 内存释放
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    return 0;
}
