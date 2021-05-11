#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* Q1: Write a CUDA kernels for Matrix Multiplication:
* a) Each row of resultant matrix is computed by one thread
* a) Each col of resultant matrix is computed by one thread
* a) Each element of resultant matrix is computed by one thread
*/
__global__ void matMultRow(int *a, int *b, int *prod, int M, int N, int P, int Q)
{
    int row = threadIdx.x;
    int prod_index, sum;

    if (N != P)
        return;

    for (int j = 0; j < Q; j++)
    {
        sum = 0;
        for (int k = 0; k < N; k++)
        {
            sum += a[row * N + k] * b[k * Q + j];
        }

        prod_index = row * Q + j;
        prod[prod_index] = sum;
    }
}

__global__ void matMultCol(int *a, int *b, int *prod, int M, int N, int P, int Q)
{
    int col = threadIdx.x;
    int prod_index, sum;

    if (N != P)
        return;

    for (int i = 0; i < M; i++)
    {
        sum = 0;
        for (int k = 0; k < N; k++)
        {
            sum += a[i * N + k] * b[k * Q + col];
        }

        prod_index = i * Q + col;
        prod[prod_index] = sum;
    }
}

__global__ void matMultElement(int *a, int *b, int *prod, int M, int N, int P, int Q)
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    int prod_index = row * N + col;

    if (N != P)
        return;

    int sum = 0;
    for (int k = 0; k < N; k++)
        sum += a[row * N + k] * b[k * Q + col];

    prod[prod_index] = sum;
}

int main()
{
    int M = 1, N = 2, P = 2, Q = 3;

    // host copies of matrices a, b
    int a[M][N] = {1, 2};
    int b[P][Q] = {{1, 2, 3}, {4, 5, 6}};

    // Separate arrays for the results of the 3 different kernel calls
    int prod1[M][Q];
    int prod2[M][Q];
    int prod3[M][Q];

    // device copies of variables a, b & prod
    int *d_a, *d_b, *d_prod;
    int sizeA = M * N * sizeof(int);
    int sizeB = P * Q * sizeof(int);
    int sizeProd = M * Q * sizeof(int);

    // Allocate space for device copies of a, b, prod
    cudaMalloc((void **)&d_a, sizeA);
    cudaMalloc((void **)&d_b, sizeB);
    cudaMalloc((void **)&d_prod, sizeProd);

    // Copy inputs to device
    cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);

    // Launch kernels on GPU:
    cudaError err;

    // a) A thread for each row
    matMultRow<<<1, M>>>(d_a, d_b, d_prod, M, N, P, Q);
    err = cudaMemcpy(&prod1, d_prod, sizeProd, cudaMemcpyDeviceToHost);

    // b) A thread for each col
    matMultCol<<<1, Q>>>(d_a, d_b, d_prod, M, N, P, Q);
    err = cudaMemcpy(&prod2, d_prod, sizeProd, cudaMemcpyDeviceToHost);

    // c) A thread for each element
    dim3 dimBlock(M, Q, 1);
    matMultElement<<<1, dimBlock>>>(d_a, d_b, d_prod, M, N, P, Q);
    err = cudaMemcpy(&prod3, d_prod, sizeProd, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    int i, j;
    printf("Matrix A (MxN): %d x %d\n", M, N);
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%d ", a[i][j]);
        printf("\n");
    }

    printf("\nMatrix B (PxQ): %d x %d\n", P, Q);
    for (i = 0; i < P; i++)
    {
        for (j = 0; j < Q; j++)
            printf("%d ", b[i][j]);
        printf("\n");
    }

    printf("\nOne thread per row:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < Q; j++)
            printf("%d ", prod1[i][j]);
        printf("\n");
    }

    printf("\nOne thread per col:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < Q; j++)
            printf("%d ", prod2[i][j]);
        printf("\n");
    }

    printf("\nOne thread per element:\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < Q; j++)
            printf("%d ", prod3[i][j]);
        printf("\n");
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_prod);
    return 0;
}