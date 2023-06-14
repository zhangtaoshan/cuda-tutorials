#include "../common/utils.h"

// 输入矩阵维度
#define INPUT_M 200
#define INPUT_N 700
#define INPUT_L 500
// 线程数和block数
#define THREADS 512
#define BLOCKS 4
// 每个网格内使用的线程数
#define NUM_THREADS_IN_ZONE 32


__global__ void matrix_multiplication_gpu_4(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    // 矩阵a和矩阵b的共享内存
    __shared__ DATATYPE matA[NUM_THREADS_IN_ZONE][NUM_THREADS_IN_ZONE];
    __shared__ DATATYPE matB[NUM_THREADS_IN_ZONE][NUM_THREADS_IN_ZONE];
    // 当前线程所在列
    const int tid_col = threadIdx.x;
    // 当前线程所在行
    const int tid_row = threadIdx.y;
    // 当前块所在列
    const int bid_col = blockIdx.x * NUM_THREADS_IN_ZONE;
    // 当前块所在行
    const int bid_row = blockIdx.y * NUM_THREADS_IN_ZONE;
    double results = 0.0;
    // 按照块大小划分矩阵并给共享内存区域赋值，沿着公共方向l
    for (int j = 0; j < l; j += NUM_THREADS_IN_ZONE)
    {
        // 往a的共享内存写数据
        if (tid_row + bid_row < m && tid_col + j < l)
        {
            // a=(m,l)
            matA[tid_row][tid_col] = a[(tid_row + bid_row) * l + (tid_col + j)];
        }
        // 越界处理
        else 
        {
            matA[tid_row][tid_col] = 0;
        }
        // 往b的共享内存写数据
        if (tid_row + j < l && tid_col + bid_col < n)
        {
            // b=(l,n)	
            matB[tid_row][tid_col] = b[(tid_row + j) * n + (tid_col + bid_col)];
        }
        else 
        {
            matB[tid_row][tid_col] = 0;
        }
        __syncthreads();
        // 在1个网格内作矩阵乘法
        for (int i = 0; i < NUM_THREADS_IN_ZONE; ++i)
        {
            results += matA[tid_row][i] * matB[i][tid_col];
        }
        __syncthreads();
    }
    if (tid_row + bid_row < m && tid_col + bid_col < n)
    {
        // c=(m,n)
	    c[(tid_row + bid_row) * n + (tid_col + bid_col)] = results;
    }
}


__global__ void matrix_multiplication_gpu_5(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    // dynamic shared memory of matrix A and matrix B
    __shared__ DATATYPE matA[NUM_THREADS_IN_ZONE][NUM_THREADS_IN_ZONE];
    __shared__ DATATYPE matB[NUM_THREADS_IN_ZONE][NUM_THREADS_IN_ZONE];
    // column NUM_THREADS_IN_ZONE
    const int tid_col = threadIdx.y;
    // row NUM_THREADS_IN_ZONE
    const int tid_row = threadIdx.x;
    // column blocks
    const int bid_col = blockIdx.y * NUM_THREADS_IN_ZONE;
    // row blocks
    const int bid_row = blockIdx.x * NUM_THREADS_IN_ZONE;
    double results = 0.0;
    for (int j = 0; j < l; j += NUM_THREADS_IN_ZONE)
    {
        // a=(m,l)
        matA[tid_row][tid_col] = a[(tid_row + bid_row) * l + tid_col + j];
        // b=(l,n)	
        matB[tid_row][tid_col] = b[(tid_row + j) * n + tid_col + bid_col];
            __syncthreads();
        // do matrix multiplication in one zone
        for (int i = 0; i < NUM_THREADS_IN_ZONE; ++i)
        {
            results += matA[tid_row][i] * matB[i][tid_col];
        }
        __syncthreads();
    }
    if (tid_row + bid_row < m && tid_col + bid_col < n)
    {
	    c[(tid_row + bid_row) * n + tid_col + bid_col] = results;
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
    cudaError_t err = cudaGetLastError();
    if (err != 0)
    {
        printf("error in cudaMalloc: %s\n", cudaGetErrorString(err));
    }
    DATATYPE* d_c = NULL;
    cudaMalloc((void**)&d_c, size_c);
    // 数据拷贝
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != 0)
    {
        printf("error in cudaMemcpy: %s\n", cudaGetErrorString(err));
    }
    // 矩阵a的1行放入共享内存中
    {
        // 定义启动核函数的参数
        int bx = (INPUT_M + NUM_THREADS_IN_ZONE - 1) / NUM_THREADS_IN_ZONE;
        int by = (INPUT_M + NUM_THREADS_IN_ZONE - 1) / NUM_THREADS_IN_ZONE;
        dim3 blocksPerGrid(bx, by, 1);
        dim3 threadsPerBlock(NUM_THREADS_IN_ZONE, NUM_THREADS_IN_ZONE, 1);
        matrix_multiplication_gpu_4<<<blocksPerGrid, threadsPerBlock>>>(
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
    // 在上述基础上判断移除
    {
        // 定义启动核函数的参数
        int bx = (INPUT_M + NUM_THREADS_IN_ZONE - 1) / NUM_THREADS_IN_ZONE;
        int by = (INPUT_M + NUM_THREADS_IN_ZONE - 1) / NUM_THREADS_IN_ZONE;
        dim3 blocksPerGrid(bx, by, 1);
        dim3 threadsPerBlock(NUM_THREADS_IN_ZONE, NUM_THREADS_IN_ZONE, 1);
        matrix_multiplication_gpu_5<<<blocksPerGrid, threadsPerBlock>>>(
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
