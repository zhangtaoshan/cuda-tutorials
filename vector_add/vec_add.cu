#include "common/utils.h"


#define NUM_INPUT 100


// cpu
void vector_add_serial(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    for (int i = 0; i < NUM_INPUT; ++i)
    {
        c[i] = a[i] + b[i];
    }
}


// single block, single thread
__global__ void vector_add_gpu_1(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    for (int i = 0; i < NUM_INPUT; ++i)
    {
        c[i] = a[i] + b[i];
    }
}


// single block, multiple threads
__global__ void vector_add_gpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    // the index of current thread
    int tid = threadIdx.x;
    // the step of threads in block
    const int t_n = blockDim.x;
    for (; tid < NUM_INPUT; tid += t_n)
    {
        c[tid] = a[tid] + b[tid];
    }
}


// multiple blocks, multiple threads
__global__ void vector_add_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    // the index of current thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // the step of threads in grid
    const int t_n = gridDim.x * blockDim.x;
    for (; tid < NUM_INPUT; tid += t_n)
    {
        c[tid] = a[tid] + b[tid];
    }
}


int main()
{
    int input_flag = 0;
    printf("input number to call different function: ");
    scanf("%d", &input_flag);
    srand(20);
    bool is_cuda = input_flag > 0;
    // the size of the memory of input
    size_t size = sizeof(DATATYPE) * NUM_INPUT;
    // malloc in cpu
    DATATYPE* h_a = (DATATYPE*)malloc(size);
    DATATYPE* h_b = (DATATYPE*)malloc(size);
    DATATYPE* h_c = (DATATYPE*)malloc(size);
    // initialize input vector
    for (int i = 0; i < NUM_INPUT; ++i)
    {
	h_a[i] = rand() / (float)RAND_MAX;
	h_b[i] = rand() / (float)RAND_MAX;
    }
    // allocate the device for input and output if necessary
    DATATYPE* d_a = NULL;
    DATATYPE* d_b = NULL;
    DATATYPE* d_c = NULL;
    // call CUDA kernel
    cudaError_t err;
    if (is_cuda)
    {
        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("Error in cudaMalloc: %s\n", cudaGetErrorString(err));
            return -1;
        }
        // memory copy
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
        err = cudaGetLastError();
        if (err != 0)
        {
            printf("Error in cudaMemcpy: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }
    if (input_flag == 0)
    {
        vector_add_serial(h_a, h_b, h_c);
    }
    else if (input_flag == 1)
    {
        vector_add_gpu_1<<<1, 1>>>(d_a, d_b, d_c);        
    }
    else if (input_flag == 2)
    {
        vector_add_gpu_2<<<1, 2>>>(d_a, d_b, d_c);        
    }
    else if (input_flag == 3)
    {
        vector_add_gpu_3<<<2, 2>>>(d_a, d_b, d_c);        
    }
    else 
    {
        printf("Error input number.\n");
        return -1;
    }
    if (is_cuda)
    {
        // memory copy
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
        // memory delete
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    print_vector(h_c, NUM_INPUT);
    return 0;
}

