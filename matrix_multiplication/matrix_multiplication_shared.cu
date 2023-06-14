#include "../common/utils.h"

// 输入矩阵维度
#define INPUT_M 200
#define INPUT_N 700
#define INPUT_L 500
// 线程数和block数
#define THREADS 512
#define BLOCKS 4


// TODO: 访存对齐情况，采用填充的办法
__global__ void matrix_multiplication_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    // 动态共享内存
    extern __shared__ DATATYPE data[];
    const int tidx = threadIdx.x;
    const int row = blockIdx.x;
    // 将a的1行放入共享内存，即长度为l的数组，每1行的处理使用1个block
    for (int i = tidx; i < l; i += blockDim.x)
    {
        // a=(m,l)，每1个block里面有自己的共享内存即a的1行数据
        data[i] = a[row * l + i];
    }
    __syncthreads();
    double temp = 0.0;
    // 迭代计算结果c的1列，这里使用到了所有线程
    for (int j = tidx; j < n; j += blockDim.x)
    {
        temp = 0.0;
        for (int i = 0; i < l; ++i)
        {
            // a=(m,l), b=(l,n)
            temp += data[i] * b[i * n + j];
        }
        // c=(m,n)，1个block写入输出矩阵的1行
        c[row * n + j] = temp;
    }
    __syncthreads();
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
        dim3 blocksPerGrid(INPUT_M, 1, 1);
        dim3 threadsPerBlock(THREADS, 1, 1);
        matrix_multiplication_gpu_3<<<blocksPerGrid, threadsPerBlock, sizeof(DATATYPE) * INPUT_L>>>(
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
