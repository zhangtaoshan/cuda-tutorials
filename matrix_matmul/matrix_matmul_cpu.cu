#include "../common/utils.h"

// 输入矩阵维度
#define INPUT_M 2
#define INPUT_N 7
#define INPUT_L 5


// 相比于原普通矩阵乘法，由于在取b矩阵的列元素时数据不连续，这里交换内部两层循环并做适当处理
void matrix_matmul_cpu_1(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    // 初始化结果矩阵
    double temp = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
	    c[i] = 0.0;
    }
    for (int i = 0; i < m; ++i)
    {
        for (int k = 0; k < l; ++k)
        {
            // 将a的(i,k)与b的第k行相乘，得到c的第i行
            // 此时得到的c为临时值并非最终的结果
            temp = a[i * l + k];
            for (int j = 0; j < n; ++j)
            {
                c[i * n + j] += temp * b[k * n + j];
            }
        }
    }
}


// 在原普通矩阵乘法中，由于b不连续所有会降低访存效率，现将b转置而每次取b矩阵的一行
void matrix_matmul_cpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c, int m, int n, int l)
{
    // 将矩阵b转置
    DATATYPE* b_t = (DATATYPE*)malloc(sizeof(DATATYPE) * l * n);
    for (int i = 0; i < l; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            b_t[j * l + i] = b[i * n + j];
        }
    }
    // 矩阵a的i行乘矩阵b_t的j行，列数都是l
    double temp = 0.0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            temp = 0.0;
            for (int k = 0; k < l; ++k)
            {
        	    temp += a[i * l + k] * b_t[j * l + k];
            }
            c[i * n + j] = temp;
        }
    }
    free(b_t);
}


int main()
{
    size_t size_a = sizeof(DATATYPE) * INPUT_M * INPUT_L;
    size_t size_b = sizeof(DATATYPE) * INPUT_L * INPUT_N;
    size_t size_c = sizeof(DATATYPE) * INPUT_M * INPUT_N;
    DATATYPE* h_a = (DATATYPE*)malloc(size_a);
    DATATYPE* h_b = (DATATYPE*)malloc(size_b);
    // 初始化输入矩阵
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
    matrix_matmul_baseline(h_a, h_b, baseline, INPUT_M, INPUT_N, INPUT_L);
    // 交换内部两层循环优化数据访问
    {
        matrix_matmul_cpu_1(h_a, h_b, h_c, INPUT_M, INPUT_N, INPUT_L);
        check_matrix(baseline, h_c, INPUT_M, INPUT_N);
    }
    // 将矩阵b转置
    {
        matrix_matmul_cpu_2(h_a, h_b, h_c, INPUT_M, INPUT_N, INPUT_L);
        check_matrix(baseline, h_c, INPUT_M, INPUT_N);
    }
    // 内存释放
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
