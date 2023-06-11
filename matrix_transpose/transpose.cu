#include <../common/utils.h>

#define DATATYPE float
#define BLOCKS 16
#define THREADS 64
#define INPUT_M 256
#define INPUT_N 512


void matrix_transposition_serial_2(DATATYPE* a, DATATYPE* b, int m, int n)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            b[j * m + i] = a[i * n + j];
        }
    }
}


__global__ void matrix_transposition_gpu_1d_1(DATATYPE* a, DATATYPE* b, int m, int n)
{ 
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
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


__global__ void matrix_transposition_gpu_1d_2(DATATYPE* a, DATATYPE* b, int m, int n)
{ 
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    while (tidx < n)
    {
        while (bidx < m)
        {
            b[tidx * m + bidx] = a[bidx * n + tidx];
            bidx += gridDim.x;
        }
        tidx += blockDim.x;
    }
}


__global__ void matrix_transposition_gpu_2d_1(DATATYPE* a, DATATYPE* b, int m, int n)
{
    const int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    if (xIndex < n && yIndex < m)
    {
        b[yIndex + m * xIndex] = a[xIndex + n * yIndex];
    }
}


__global__ void matrix_transposition_gpu_2d_2(DATATYPE* a, DATATYPE* b, int m, int n)
{
    __shared__ DATATYPE tmp[BLOCKS][BLOCKS + 1];
    // put the elements from a to shared memory
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex < n && yIndex < m)
    {
        tmp[threadIdx.y][threadIdx.x] = a[yIndex * n + xIndex];
    }
    __syncthreads();
    xIndex = blockIdx.y * BLOCKS + threadIdx.x;
    yIndex = blockIdx.x * BLOCKS + threadIdx.y;
    if (xIndex < m && yIndex < n)
    {
        b[yIndex * m + xIndex] = tmp[threadIdx.x][threadIdx.y];
    }
}


int main()
{
    int input_flag = 0;
    printf("input number to call different function: ");
    scanf("%d", &input_flag);
    srand(20);
    size_t size_a = sizeof(DATATYPE) * INPUT_M * INPUT_N;
    DATATYPE* h_a = (DATATYPE*)malloc(size_a);
    // initialize input vector
    for (int i = 0; i < INPUT_M; ++i)
    {
        for (int j = 0; j < INPUT_N; ++j)
        {
            h_a[i * INPUT_N + j] = rand() / (DATATYPE)RAND_MAX;
        }
    }
    // allocate the device for input and output
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size_a);
    cudaError_t err = cudaGetLastError();
    if (err != 0)
    {
        printf("CUDA malloc a error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    size_t size_b = sizeof(DATATYPE) * INPUT_N * INPUT_M;
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size_b);
    err = cudaGetLastError();
    if (err != 0)
    {
        printf("CUDA malloc b error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    DATATYPE* h_b = (DATATYPE*)malloc(size_b);
    if (err != 0)
    {
        printf("CUDA malloc b error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // memory copy
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    if (input_flag == 0)
    {
        matrix_transposition_serial_1(h_a, h_b, INPUT_M, INPUT_N);
    }
    else if (input_flag == 1)
    {
        matrix_transposition_serial_2(h_a, h_b, INPUT_M, INPUT_N);
    }
    else if (input_flag == 2)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = BLOCKS;
        matrix_transposition_gpu_1d_1<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, INPUT_M, INPUT_N);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        cudaMemcpy(h_b, d_b, size_b, cudaMemcpyDeviceToHost);
    }
    else if (input_flag == 3)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = BLOCKS;
        matrix_transposition_gpu_1d_2<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, INPUT_M, INPUT_N);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        cudaMemcpy(h_b, d_b, size_b, cudaMemcpyDeviceToHost);
    }
    else if (input_flag == 4)
    {
        dim3 blocksPerGrid(BLOCKS, BLOCKS);
        int xThreads = (INPUT_N + BLOCKS - 1) / BLOCKS;
        int yThreads = (INPUT_M + BLOCKS - 1) / BLOCKS;
        dim3 threadsPerBlock(xThreads, yThreads);
        matrix_transposition_gpu_2d_1<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, INPUT_M, INPUT_N);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        cudaMemcpy(h_b, d_b, size_b, cudaMemcpyDeviceToHost);
    }
    else if (input_flag == 5)
    {
        dim3 threadsPerBlock(BLOCKS, BLOCKS);
        int xThreads = (INPUT_N + BLOCKS - 1) / BLOCKS;
        int yThreads = (INPUT_M + BLOCKS - 1) / BLOCKS;
        dim3 blocksPerGrid(xThreads, yThreads);
        matrix_transposition_gpu_2d_2<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, INPUT_M, INPUT_N);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        cudaMemcpy(h_b, d_b, size_b, cudaMemcpyDeviceToHost);
    }
    else
    {
        printf("Error input numer.\n");
        return -1;
    }
    print_matrix(h_b, INPUT_N, INPUT_M);
    // memory delete
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}

