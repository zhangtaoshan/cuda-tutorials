#include "../common/utils.h"

// 输入矩阵维度
#define INPUT_M 200
#define INPUT_N 700
// 线程数和block数
#define THREADS 32


// TODO: 和giagonal优化对比
__global__ void matrix_transposition_gpu_2d_1(DATATYPE* a, DATATYPE* b, int m, int n)
{
    // 针对输出矩阵，xIndex索引行，yIndex索引列，输入矩阵相反
    const int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    if (xIndex < n && yIndex < m)
    {
        // b=(n,m), a=(m,n)
        b[xIndex * m + yIndex] = a[yIndex * n + xIndex];
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
    // 使用2维线程 
    {
        // 定义启动核函数的参数
        int nbx = (INPUT_N + THREADS - 1) / THREADS;
        int nby = (INPUT_M + THREADS - 1) / THREADS;
        dim3 blocksPerGrid(nbx, nby, 1);
        dim3 threadsPerBlock(THREADS, THREADS, 1);
        matrix_transposition_gpu_2d_1<<<blocksPerGrid, threadsPerBlock>>>(
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
