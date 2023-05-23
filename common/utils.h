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
