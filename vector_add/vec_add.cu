#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATATYPE float


void vector_add_serial(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}


// single block, single thread
__global__ void vector_add_gpu_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}


// single block, multiple threads
__global__ void vector_add_gpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n)
{
    int tid = threadIdx.x;
    const int t_n = blockDim.x;
    for (; tid < n; tid += t_n)
    {
        c[tid] = a[tid] + b[tid];
    }
}


// multiple blocks, multiple threads
__global__ void vector_add_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n)
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
    int n = 50000;
    size_t size = sizeof(DATATYPE) * n;
    DATATYPE* h_a = (DATATYPE*)malloc(size);
    DATATYPE* h_b = (DATATYPE*)malloc(size);
    DATATYPE* h_c = (DATATYPE*)malloc(size);
    // initialize input vector
    for (int i = 0; i < n; ++i)
    {
	h_a[i] = rand() / (float)RAND_MAX;
	h_b[i] = rand() / (float)RAND_MAX;
    }
    // allocate the device for input and output
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size);
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size);
    DATATYPE* d_c = NULL;
    cudaMalloc((void**)&d_c, size);
    // memory copy
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    // launch the cuda kernel
    int threadsPerBlock = 4;
    int blocksPerGrid = 2;
    vector_add_gpu_3<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    // memory copy
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    // memory delete
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // return
    return 0;
}

