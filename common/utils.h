#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATATYPE float


void print_vector(DATATYPE* v, int n)
{
    for (int i = 0; i < n; ++i)
    {
        printf("%f ", v[i]);
    }
    printf("\n");
}


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


void vector_add_baseline(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}


void vector_dot_baseline(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n)
{
    double temp = 0.0;
    for (int i = 0; i < n; ++i)
    {
        temp += a[i] * b[i];
    }
    *c = temp;
}


void matrix_matmul_baseline(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
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


int check_value(DATATYPE* a, DATATYPE* b)
{
    // 原子操作使得结果精度较低
    if (abs(*a - *b) > 1e-3)
    {
        printf("bad accuracy.\n");
        return -1;
    }
    printf("ok.\n");
    return 0;
}


int check_vector(DATATYPE* a, DATATYPE* b, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (abs(a[i] - b[i]) > 1e-4)
        {
            printf("bad accuracy.\n");
            return -1;
        }
    }
    printf("ok.\n");
    return 0;
}


int check_matrix(DATATYPE* a, DATATYPE* b, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            // 1e-4通不过会有精度损失
            if (abs(a[i * n + j] - b[i * n + j]) > 1e-3)
            {
                printf("bad accuracy.\n");
                return -1;
            }
        }
    }
    printf("ok.\n");
    return 0;  
}
