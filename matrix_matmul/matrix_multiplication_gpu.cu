#include "../common/utils.h"

// 输入矩阵维度
#define INPUT_M 200
#define INPUT_N 700
#define INPUT_L 500
// 线程数和block数
#define THREADS 512
#define BLOCKS 4


__global__ void matrix_multiplication_gpu_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    // 每个线程处理a的1行和b的1列的内积运算
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    // 线程在grid内的索引，每个线程计算a的1行和b的1列
    const int idx = bidx * blockDim.x + tidx;
    // 行索引
    const int row = idx / n;
    // 列索引
    const int col = idx % n;
    if (row < m && col < n)
    {
        double temp = 0.0;
        for (int i = 0; i < l; ++i)
        {
            // a=(m,l), b=(l,n)
            temp += a[row * l + i] * b[i * n + col];
        }
	// c=(m,n)
	c[row * n + col] = temp;
    }
}


__global__ void matrix_multiplication_gpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    double tmp = 0.0;
    // 每个block计算输出矩阵的1行
    for (; bidx < m; bidx += gridDim.x)
    {
        for (; tidx < n; tidx += blockDim.x)
        {
            tmp = 0.0;
            for (int i = 0; i < l; ++i)
            {
                // a=(m,l), b=(l,n)
                tmp += a[bidx * l + i] * b[i * n + tidx];
            }
            // c=(m,n)
            c[bidx * n + tidx] = tmp;
        }
    }
}


int main()
{
    size_t size_a = sizeof(DATATYPE) * INPUT_M * INPUT_L;
    size_t size_b = sizeof(DATATYPE) * INPUT_L * INPUT_N;
    size_t size_c = sizeof(DATATYPE) * INPUT_M * INPUT_N;
    DATATYPE* h_a = (DATATYPE*)malloc(size_a);
    DATATYPE* h_b = (DATATYPE*)malloc(size_b);
    // 输入初始化
    for (int i = 0; i < INPUT_M; ++i)
    {
        for (int j = 0; j < INPUT_L; ++j)
        {
            h_a[i * INPUT_L + j] = rand() / (DATATYPE)RAND_MAX;
        }
    }
    for (int i = 0; i < INPUT_L; ++i)
    {
        for (int j = 0; j < INPUT_N; ++j)
        {
            h_b[i * INPUT_N + j] = rand() / (DATATYPE)RAND_MAX;
        }
    }
    DATATYPE* h_c = (DATATYPE*)malloc(size_c);
    // baseline
    DATATYPE* baseline = (DATATYPE*)malloc(size_c);
    matrix_multiplication_baseline(h_a, h_b, baseline, INPUT_M, INPUT_N, INPUT_L);
    // 分配设备上的内存
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size_a);
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size_b);
    DATATYPE* d_c = NULL;
    cudaError_t err = cudaGetLastError();
    if (err != 0)
    {
        printf("error in cudaMalloc: %s\n", cudaGetErrorString(err));
    }
    cudaMalloc((void**)&d_c, size_c);
    // 数据拷贝
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != 0)
    {
        printf("error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    }
    // 使用grid内所有线程计算
    {
        // 定义启动核函数的参数
        dim3 blocksPerGrid(((INPUT_M + THREADS - 1) / THREADS) * INPUT_M, 1, 1);
        dim3 threadsPerBlock(THREADS, 1, 1);
        matrix_multiplication_gpu_1<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c, INPUT_M, INPUT_N, INPUT_L);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("error in kernel forward: %s\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
        check_matrix(baseline, h_c, INPUT_M, INPUT_N);
    }
    // 每个block计算得到输出矩阵的1行
    {
        // 定义启动核函数的参数
        dim3 blocksPerGrid(INPUT_M, 1, 1);
        dim3 threadsPerBlock(THREADS, 1, 1);
        matrix_multiplication_gpu_2<<<blocksPerGrid, threadsPerBlock>>>(
            d_a, d_b, d_c, INPUT_M, INPUT_N, INPUT_L);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("error in kernel forward: %s\n", cudaGetErrorString(err));
        }
        // 拷贝输出数据
        cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
        check_matrix(baseline, h_c, INPUT_M, INPUT_N);  
    }
    // 内存释放
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
