#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* Q1: Write a CUDA kernels for Matrix Addition:
* a) Each row of resultant matrix is computed by one thread
* a) Each col of resultant matrix is computed by one thread
* a) Each element of resultant matrix is computed by one thread
*/
__global__ void matAddRow(int *a, int *b, int *sum, int M, int N)
{
    int row = threadIdx.x;
    if (row >= M)
        return;

    int index;
    for (int j = 0; j < N; j++)
    {
        index = row * N + j;
        sum[index] = a[index] + b[index];
    }
}

__global__ void matAddCol(int *a, int *b, int *sum, int M, int N)
{
    int col = threadIdx.x;
    if (col >= N)
        return;

    int index;
    for (int i = 0; i < M; i++)
    {
        index = i * N + col;
        sum[index] = a[index] + b[index];
    }
}

__global__ void matAddElement(int *a, int *b, int *sum, int M, int N)
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    int index = row * N + col;

    if (row < M && col < N)
        sum[index] = a[index] + b[index];
}

int main()
{
    int M = 3, N = 3;
    // host copies of matrices a, b
    int a[M][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int b[M][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    // Separate arrays for the results of the 3 different kernel calls
    int sum1[M][N];
    int sum2[M][N];
    int sum3[M][N];

    // device copies of variables a, b & sum
    int *d_a, *d_b, *d_sum;
    int size = M * N * sizeof(int);

    // Allocate space for device copies of a, b, sum
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_sum, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernels on GPU:
    cudaError err;

    // a) A thread for each row
    matAddRow<<<1, M>>>(d_a, d_b, d_sum, M, N);
    err = cudaMemcpy(&sum1, d_sum, size, cudaMemcpyDeviceToHost);

    // a) A thread for each col
    matAddCol<<<1, N>>>(d_a, d_b, d_sum, M, N);
    err = cudaMemcpy(&sum2, d_sum, size, cudaMemcpyDeviceToHost);

    // c) A thread for each element
    dim3 dimBlock(M, N, 1);
    matAddElement<<<1, dimBlock>>>(d_a, d_b, d_sum, M, N);
    err = cudaMemcpy(&sum3, d_sum, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    int i, j;
    printf("One thread per row:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d ", sum1[i][j]);
        printf("\n");
    }
    printf("\nOne thread per col:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d ", sum2[i][j]);
        printf("\n");
    }
    printf("\nOne thread per element:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d ", sum3[i][j]);
        printf("\n");
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_sum);
    return 0;
}