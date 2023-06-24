#include "../common/utils.h"

// 输入矩阵维度
#define INPUT_M 200
#define INPUT_N 700
// 线程数和block数
#define BLOCK_DIM 16


__global__ void matrix_transposition_gpu_2d_2(DATATYPE* a, DATATYPE* b, int m, int n)
{
    __shared__ DATATYPE tmp[BLOCK_DIM][BLOCK_DIM + 1];
    // put the elements from a to shared memory
    int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if (xIndex < n && yIndex < m)
    {
        // a=(m,n)
        tmp[threadIdx.y][threadIdx.x] = a[yIndex * n + xIndex];
    }
    __syncthreads();
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if (xIndex < m && yIndex < n)
    {
        // b=(n,m)
        b[yIndex * m + xIndex] = tmp[threadIdx.x][threadIdx.y];
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
    {
        // 定义启动核函数的参数
        int nbx = (INPUT_N + BLOCK_DIM - 1) / BLOCK_DIM;
        int nby = (INPUT_M + BLOCK_DIM - 1) / BLOCK_DIM;
        dim3 blocksPerGrid(nbx, nby, 1);
        dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM, 1);
        matrix_transposition_gpu_2d_2<<<blocksPerGrid, threadsPerBlock>>>(
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
