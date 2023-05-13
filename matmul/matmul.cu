#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATATYPE float
#define BLOCKS 4
#define THREADS 4
#define INPUT_M 8
#define INPUT_N 16
#define INPUT_K 32


void print_matrix(DATATYPE* v, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
	for (int j = 0; j < n; ++j)
	{
	    printf("%f ", v[i * n + j]);
	}
	printf("\n");
    }
}


void matrix_multiplication_serial_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    double temp = 0.0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
	{
	    temp = 0.0;
	    for (int k = 0; k < l; ++k)
	    {
		temp += a[i * l + k] * b[k * n + j];
	    }
	    c[i * n + j] = temp;
	}
    }
}


void matrix_multiplication_serial_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    double temp = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
	c[i] = 0.0;
    }
    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < l; ++k)
	{
	    temp = a[i * l + k];
	    for (int j = 0; j < n; ++j)
	    {
	        c[i * n + j] += temp * b[k * n + j];
	    }
	}
    }
}


void matrix_multiplication_serial_3(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    double temp = 0.0;
    DATATYPE* b_t = (DATATYPE*)malloc(sizeof(DATATYPE) * l * n);
    for (int i = 0; i < l; ++i)
    {
	for (int j = 0; j < n; ++j)
	{
	    b_t[i * l + j] = b[j * n + i];
	}
    }
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
	{
	    temp = 0.0;
	    for (int k = 0; k < l; ++k)
	    {
		temp += a[i * l + k] * b_t[j * n + k];
	    }
	    c[i * n + j] = temp;
	}
    }
    free(b_t);
}


int main()
{
    int input_flag = 0;
    printf("input number to call different function: ");
    scanf("%d", &input_flag);
    srand(20);
    size_t size_a = sizeof(DATATYPE) * INPUT_M * INPUT_K;
    size_t size_b = sizeof(DATATYPE) * INPUT_K * INPUT_N;
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
	    h_a[i * INPUT_N + j] = rand() / (DATATYPE)RAND_MAX;
	}
    }
    // allocate the device for input and output
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size_a);
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size_b);
    DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE) * INPUT_M * INPUT_N);
    DATATYPE* d_c = NULL;
    cudaMalloc((void**)d_c, sizeof(DATATYPE) * INPUT_M * INPUT_N);
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
    cudaMemcpy(h_c, d_c, sizeof(DATATYPE) * INPUT_M * INPUT_N, cudaMemcpyDeviceToHost);
    print_matrix(h_c, INPUT_M, INPUT_N);
    cudaFree(d_c);
    // memory delete
    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}

