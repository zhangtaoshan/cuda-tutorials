#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATATYPE float


void print_vector(DATATYPE* v, int n)
{
    for (int i = 0; i < n; ++i)
    {
        printf("%f", v[i]);
    }
}


void print_matrix(DATATYPE* v, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            printf("%f", v[i * n + j]);
        }
        printf("\n");
    }
}


double vector_add_baseline(DATATYPE* a, DATATYPE* b, int n)
{
    DATATYPE* c = (DATATYPE*)malloc(sizeof(DATATYPE) * n);
    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
    for (int i = 0; i < n; ++i)
    {
        printf("%f ", c[i]);
    }
    printf("\n");
    free(c);
}


double vector_dot_baseline(DATATYPE* a, DATATYPE* b, int n)
{
    double c = 0.0;
    for (int i = 0; i < n; ++i)
    {
        c += a[i] * b[i];
    }
    printf("baseline results: %f\n", c);
}


void matrix_transposition_baseline(DATATYPE* a, int m, int n)
{
    DATATYPE* b = (DATATYPE*)malloc(sizeof(DATATYPE) * m * n);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            b[j * m + i] = a[i * n + j];
        }
    }
    print_matrix(b, m, n);
}


void matrix_matmul_baseline(DATATYPE* a, DATATYPE* b, int m, int n, int l)
{
    DATATYPE* c = (DATATYPE*)malloc(sizeof(DATATYPE) * m * n);
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
    print_matrix(c, m, n);
}
