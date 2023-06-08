#include <common/utils.h>


#define BLOCKS 4
#define THREADS 512
#define NUM_INPUT 2000

__device__ unsigned int lockcount = 0;


// 单个线程，类似于CPU的串行执行
void vector_add_serial(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    double temp = 0.0;
    for (int i = 0; i < NUM_INPUT; ++i)
    {
        temp += a[i] * b[i];
    }
    *c = temp;
}


// 使用单个block多个线程，使用分散规约
__global__ void vector_dot_product_gpu_1(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    // 共享内存大小定义为线程数
    __shared__ DATATYPE tmp[THREADS];
    const int tidx = threadIdx.x;
    // 如果当前线程数少于数据，则需要线程作第多次计算，定义步长
    const int t_n = blockDim.x;
    // 每个线程有自己的变量
    double temp = 0.0;
    // 每次处理线程数个元素，如索引为0的线程的temp已经累加了0, 0+THREADS, 0+2*THREADS...的元素
    for (int tid = tidx; tid < NUM_INPUT; tid += t_n)
    {
	    temp += a[tid] * b[tid];
    }
    // 为共享空间的每个位置赋值
    tmp[tidx] = temp;
    __syncthreads();
    // 对THREADS个元素规约
    int i = 2, j = 1;
    // 进行log(THREADS)次计算
    while (i <= THREADS)
    {
        if ((tidx % i) == 0)
	    {
            // 后一个规约到当前
	        tmp[tidx] += tmp[tidx + j];
	    }
        __syncthreads();
        i *= 2;
        j *= 2;
    }
    // 规约完成后第一个位置的值即为最终答案
    if (tidx == 0)
    {
	    c[0] = tmp[0];
    }
}


// 使用单个block多个线程，使用低线程规约
__global__ void vector_dot_product_gpu_2(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    __shared__ DATATYPE tmp[NUM_INPUT];
    int tidx = threadIdx.x;
    const int t_n = blockDim.x;
    double temp = 0.0;
    for (int tid = tidx; tid < NUM_INPUT; tid += t_n)
    {
	    temp += a[tid] * b[tid];
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = NUM_INPUT / 2;
    while (i != 0)
    {
        // 把线程分为小于i和大于i的两部分
	if (tidx < i)
	{
            // 后面部分的元素以相同步长规约到前面元素
	    tmp[tidx] += tmp[tidx + i];
	}
	__syncthreads();
	i /= 2;
    }
    if (tidx == 0)
    {
	c[0] = tmp[0];
    }
}


// 使用多个block，最后规约在CPU上做，每个block内部发生了和gpu_2中相同的操作
__global__ void vector_dot_product_gpu_3(DATATYPE* a, DATATYPE* b, DATATYPE* c_temp)
{
    __shared__ DATATYPE tmp[THREADS];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    for (; tid < NUM_INPUT; tid += t_n)
    {
	temp += a[tid] * b[tid];
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = NUM_INPUT / 2;
    while (i != 0)
    {
	if (tidx < i)
	{
	    tmp[tidx] += tmp[tidx + i];
	}
	__syncthreads();
	i /= 2;
    }
    if (tidx == 0)
    {
	c_temp[bidx] = tmp[0];
    }
}


// 使用多个block，最后规约在GPU上做
__global__ void vector_dot_product_gpu_4(DATATYPE* c_temp, DATATYPE* c)
{
    // 共享内存大小声明为block的数量
    __shared__ DATATYPE tmp[BLOCKS];
    const int tidx = threadIdx.x;
    tmp[tidx] = c_temp[tidx];
    __syncthreads();
    int i = BLOCKS / 2;
    while (i != 0)
    {
	if (tidx < i)
	{
	    tmp[tidx] += tmp[tidx + i];
	}
	__syncthreads();
	i /= 2;
    }
    // all elements reduce to the first element
    if (tidx == 0)
    {
	c[0] = tmp[0];
    }
}


// use aotmic to reduce vector add
__global__ void vector_dot_product_gpu_5(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    // initialize to zero
    if ((threadIdx.x == 0) && (blockIdx.x == 0))
    {
	c[0] = 0.0;
    }
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    for (; tid < NUM_INPUT; tid += t_n)
    {
	temp += a[tid] * b[tid];
    }
    atomicAdd(c, temp);
}


// use aotmic to reduce vector add
__global__ void vector_dot_product_gpu_6(DATATYPE* a, DATATYPE* b, DATATYPE* c)
{
    // initialize to zero
    if ((threadIdx.x == 0) && (blockIdx.x == 0))
    {
	c[0] = 0.0;
    }
    __shared__ DATATYPE tmp[NUM_INPUT];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    for (; tid < NUM_INPUT; tid += t_n)
    {
	temp += a[tid] * b[tid];
    }
    tmp[tidx] = temp;
    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0)
    {
	if (tidx < i)
	{
	    tmp[tidx] += tmp[tidx + i];
	}
	__syncthreads();
	i /= 2;
    }
    if (tidx == 0)
    {
	atomicAdd(c, tmp[0]);
    }
}


__device__ void vector_dot(DATATYPE* out, DATATYPE* temp)
{
    const int tidx = threadIdx.x;
    int i = blockDim.x / 2;
    while (i != 0)
    {
	if (tidx < i)
	{
	    temp[tidx] += temp[tidx + i];
	}
	__syncthreads();
	i /= 2;
    }
    if (tidx == 0)
    {
	out[0] = temp[0];
    }
}


__global__ void vector_dot_product_gpu_7(DATATYPE* a, DATATYPE* b, DATATYPE* c, DATATYPE* c_temp)
{
    __shared__ DATATYPE tmp[NUM_INPUT];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    for (; tid < NUM_INPUT; tid += t_n)
    {
	temp += a[tid] * b[tid];
    }
    tmp[tidx] = temp;
    __syncthreads();
    vector_dot(&c_temp[blockIdx.x], tmp);
    __shared__ bool lock;
    __threadfence();
    if (tidx == 0)
    {
	unsigned int lockiii = atomicAdd(&lockcount, 1);
        lock = (lockcount == gridDim.x);
    }
    __syncthreads();
    if (lock)
    {
	tmp[tidx] = c_temp[tidx];
	__syncthreads();
	vector_dot(c, tmp);
	lockcount = 0;
    }
}


int main()
{
    int input_flag = 0;
    printf("input number to call different function: ");
    scanf("%d", &input_flag);
    srand(20);
    size_t size = sizeof(DATATYPE) * NUM_INPUT;
    DATATYPE* h_a = (DATATYPE*)malloc(size);
    DATATYPE* h_b = (DATATYPE*)malloc(size);
    // initialize input vector
    for (int i = 0; i < NUM_INPUT; ++i)
    {
	h_a[i] = rand() / (DATATYPE)RAND_MAX;
	h_b[i] = rand() / (DATATYPE)RAND_MAX;
    }
    // allocate the device for input and output
    DATATYPE* d_a = NULL;
    cudaMalloc((void**)&d_a, size);
    DATATYPE* d_b = NULL;
    cudaMalloc((void**)&d_b, size);
    // memory copy
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (input_flag == 0)
    {
	DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE));
        vector_add_serial(h_a, h_b, h_c);
	printf("result: %f\n", *h_c);
    }
    else if (input_flag == 1)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = 1;
	DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE));
        DATATYPE* d_c = NULL;
        cudaMalloc((void**)&d_c, sizeof(DATATYPE));
        vector_dot_product_gpu_1<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
        // memory copy
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
	printf("result: %f\n", *h_c);
        cudaFree(d_c);
    }
    else if (input_flag == 2)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = 1;
	DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE));
        DATATYPE* d_c = NULL;
        cudaMalloc((void**)&d_c, sizeof(DATATYPE));
        vector_dot_product_gpu_2<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
        // memory copy
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
	printf("result: %f\n", *h_c);
        cudaFree(d_c);
    }
    else if (input_flag == 3)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = BLOCKS;
	int blocknum = blocksPerGrid;
	DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE) * blocknum);
        DATATYPE* d_c = NULL;
        cudaMalloc((void**)&d_c, sizeof(DATATYPE) * blocknum);
        vector_dot_product_gpu_3<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
        // memory copy
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE) * blocknum, cudaMemcpyDeviceToHost);
	// vector reduction in cpu
	DATATYPE temp = 0.0;
	for (int i = 0; i < blocknum; ++i)
	{
	    temp += h_c[i];
	}
        printf("result: %f\n", temp);
        cudaFree(d_c);
    }
    else if (input_flag == 4)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = BLOCKS;
	int blocknum = blocksPerGrid;
	DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE));
        DATATYPE* d_c_temp = NULL;
        cudaMalloc((void**)&d_c_temp, sizeof(DATATYPE) * blocknum);
        vector_dot_product_gpu_3<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c_temp);
	// vector reduction in gpu
	DATATYPE* d_c = NULL;
        cudaMalloc((void**)&d_c, sizeof(DATATYPE));
        vector_dot_product_gpu_4<<<1, blocksPerGrid>>>(d_c_temp, d_c);
        // memory copy
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
        printf("result: %f\n", *h_c);
        cudaFree(d_c);
    }
    else if (input_flag == 5)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = BLOCKS;
	DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE));
        DATATYPE* d_c = NULL;
        cudaMalloc((void**)&d_c, sizeof(DATATYPE));
        vector_dot_product_gpu_5<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
        // memory copy
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
	// vector reduction in cpu
        printf("result: %f\n", *h_c);
        cudaFree(d_c);
    }
    else if (input_flag == 6)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = BLOCKS;
	DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE));
        DATATYPE* d_c = NULL;
        cudaMalloc((void**)&d_c, sizeof(DATATYPE));
        vector_dot_product_gpu_6<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
        // memory copy
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
	// vector reduction in cpu
        printf("result: %f\n", *h_c);
        cudaFree(d_c);
    }
    else if (input_flag == 7)
    {
        int threadsPerBlock = THREADS;
        int blocksPerGrid = BLOCKS;
	int blocknum = blocksPerGrid;
	DATATYPE* h_c = (DATATYPE*)malloc(sizeof(DATATYPE));
        DATATYPE* d_c = NULL;
        cudaMalloc((void**)&d_c, sizeof(DATATYPE));
        DATATYPE* d_c_temp = NULL;
        cudaMalloc((void**)&d_c_temp, sizeof(DATATYPE) * blocknum);
        vector_dot_product_gpu_7<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, d_c_temp);
        // memory copy
        cudaMemcpy(h_c, d_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);
	// vector reduction in cpu
        printf("result: %f\n", *h_c);
        cudaFree(d_c);
    }
    // memory delete
    cudaFree(d_a);
    cudaFree(d_b);
    // return
    return 0;
}
