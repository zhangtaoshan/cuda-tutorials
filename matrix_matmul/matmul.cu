#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATATYPE float
#define BLOCKS 4
#define THREADS 4
#define INPUT_M 8
#define INPUT_N 16
#define INPUT_K 32
#define NUM_THREADS_IN_ZONE 32


__global__ void matrix_multiplication_gpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    double tmp = 0.0;
    // each block to compute one row of output c
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


__global__ void matrix_multiplication_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    // dynamic shared memory
    extern __shared__ DATATYPE data[];
    const int tidx = threadIdx.x;
    const int row = blockIdx.x;
    // put one row of a to shared memory
    for (int i = tidx; i < l; i += blockDim.x)
    {
	// a=(m,l)
	data[i] = a[row * l + i];
    }
    __syncthreads();
    double tmp = 0.0;
    for (int j = tidx; j < l; j += blockDim.x)
    {
	tmp = 0.0;
	for (int i = 0; i < l; ++i)
	{
	    // a=(m,l), b=(l,n)
	    tmp += data[i] * b[i * n + j];
	}
	// c=(m,n)
	c[row * n + j] = tmp;
    }
}


__global__ void matrix_multiplication_gpu_4(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
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
	// write data from a to shared memory
	if (tid_row + bid_row < m && tid_col + j < l)
	{
	    // a=(m,l)
	    matA[tid_row][tid_col] = a[(tid_row + bid_row) * l + tid_col + j];
	}
	else 
	{
	    matA[tid_row][tid_col] = 0;
	}
	// write data from b to shared memory
	if (tid_row + j < l && tid_col + bid_col < n)
	{
	    // b=(l,n)	
	    matB[tid_row][tid_col] = b[(tid_row + j) * n + tid_col + bid_col];
	}
	else 
	{
	    matB[tid_row][tid_col] = 0;
	}
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
    int input_flag = 0;
    printf("input number to call different function: ");
    scanf("%d", &input_flag);
    srand(20);
    size_t size_a = sizeof(DATATYPE) * INPUT_M * INPUT_K;
    size_t size_b = sizeof(DATATYPE) * INPUT_K * INPUT_N;
    size_t size_c = sizeof(DATATYPE) * INPUT_M * INPUT_N;
    DATATYPE* h_a = (DATATYPE*)malloc(size_a);
    DATATYPE* h_b = (DATATYPE*)malloc(size_b);
    // initialize input vector
    for (int i = 0; i < INPUT_M; ++i)
    {
	for (int j = 0; j < INPUT_K; ++j)
	{
	    h_a[i * INPUT_K + j] = rand() / (DATATYPE)RAND_MAX;
	}
    }
    for (int i = 0; i < INPUT_K; ++i)
    {
	for (int j = 0; j < INPUT_N; ++j)
	{
	    h_b[i * INPUT_N + j] = rand() / (DATATYPE)RAND_MAX;
	}
    }
    // CUDA kernel params
    int threads = 1024;
    // allocate the device for input and output
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size_a);
    cudaError_t err = cudaGetLastError();
    if (err != 0)
    {
        printf("CUDA malloc a error: %s\n", cudaGetErrorString(err));
    }
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size_b);
    err = cudaGetLastError();
    if (err != 0)
    {
        printf("CUDA malloc b error: %s\n", cudaGetErrorString(err));
    }
    DATATYPE* h_c = (DATATYPE*)malloc(size_c);
    DATATYPE* d_c = NULL;
    cudaMalloc((void**)&d_c, size_c);
    err = cudaGetLastError();
    if (err != 0)
    {
        printf("CUDA malloc c error: %s\n", cudaGetErrorString(err));
    }
    // memory copy
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    if (input_flag == 0)
    {
        matrix_multiplication_serial_1(h_a, h_b, h_c, INPUT_M, INPUT_N, INPUT_K);
    }
    else if (input_flag == 1)
    {
        matrix_multiplication_serial_2(h_a, h_b, h_c, INPUT_M, INPUT_N, INPUT_K);
    }
    else if (input_flag == 2)
    {
        matrix_multiplication_serial_3(h_a, h_b, h_c, INPUT_M, INPUT_N, INPUT_K);
    }
    else if (input_flag == 3)
    {
	int threadsPerBlock = threads;
	int blocksPerGrid = (INPUT_M + threadsPerBlock - 1) / threadsPerBlock;
        matrix_multiplication_gpu_1<<<blocksPerGrid * INPUT_M, threadsPerBlock>>>(d_a, d_b, d_c, INPUT_M, INPUT_N, INPUT_K);
	cudaError_t err = cudaGetLastError();
	if (err != 0)
	{
	    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
	}
        cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    }
    else if (input_flag == 4)
    {
	int threadsPerBlock = 1024;
	int blocksPerGrid = INPUT_M;
        matrix_multiplication_gpu_2<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, INPUT_M, INPUT_N, INPUT_K);
	cudaError_t err = cudaGetLastError();
	if (err != 0)
	{
	    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
	}
        cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    }
    else if (input_flag == 5)
    {
	int threadsPerBlock = 1024;
	int blocksPerGrid = INPUT_M;
        matrix_multiplication_gpu_3<<<blocksPerGrid, threadsPerBlock, sizeof(DATATYPE) * INPUT_M>>>(d_a, d_b, d_c, INPUT_M, INPUT_N, INPUT_K);
	cudaError_t err = cudaGetLastError();
	if (err != 0)
	{
	    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
	}
        cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    }
    else if (input_flag == 6)
    {
	dim3 threadsPerBlock(NUM_THREADS_IN_ZONE, NUM_THREADS_IN_ZONE);
	int bx = (INPUT_M + NUM_THREADS_IN_ZONE - 1) / NUM_THREADS_IN_ZONE;
	dim3 blocksPerGrid(bx, bx);
        matrix_multiplication_gpu_4<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, INPUT_M, INPUT_N, INPUT_K);
	cudaError_t err = cudaGetLastError();
	if (err != 0)
	{
	    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
	}
        cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    }
    else if (input_flag == 7)
    {
	dim3 threadsPerBlock(NUM_THREADS_IN_ZONE, NUM_THREADS_IN_ZONE);
	int bx = (INPUT_M + NUM_THREADS_IN_ZONE - 1) / NUM_THREADS_IN_ZONE;
	dim3 blocksPerGrid(bx, bx);
        matrix_multiplication_gpu_5<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, INPUT_M, INPUT_N, INPUT_K);
	cudaError_t err = cudaGetLastError();
	if (err != 0)
	{
	    printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
	}
        cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
    }
    else
    {
	printf("Error input numer.\n");
	return -1;
    }
    print_matrix(h_c, INPUT_M, INPUT_N);
    // memory delete
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}

