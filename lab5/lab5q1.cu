#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#define ARR_LEN 12

/*
    Q1. Vector addition using the same kernel with 3 different launch configurations. 
*/

__global__ void add(int *a, int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    // host copies of variables a, b & c
    int a[ARR_LEN] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int b[ARR_LEN] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // Separate arrays for the results of the 3 different kernel calls
    int c1[ARR_LEN];
    int c2[ARR_LEN];
    int c3[ARR_LEN];

    // device copies of variables a, b & c
    int *d_a, *d_b, *d_c;

    int size = ARR_LEN * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    cudaError err;

    // 1a) Grid size as N
    add<<<ARR_LEN, 1>>>(d_a, d_b, d_c, ARR_LEN);
    err = cudaMemcpy(&c1, d_c, size, cudaMemcpyDeviceToHost);

    // 1b) N threads within a block
    add<<<1, ARR_LEN>>>(d_a, d_b, d_c, ARR_LEN);
    err = cudaMemcpy(&c2, d_c, size, cudaMemcpyDeviceToHost);

    // 1c) Keep the num of threads per block as 256, vary num of blocks to handle N elements.
    add<<<ceil(ARR_LEN / 256), 256>>>(d_a, d_b, d_c, ARR_LEN);
    err = cudaMemcpy(&c3, d_c, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("\na) Kernel 1: ");
    for (int i = 0; i < ARR_LEN; i++)
        printf("%d, ", c1[i]);

    printf("\nb) Kernel 2: ");
    for (int i = 0; i < ARR_LEN; i++)
        printf("%d, ", c2[i]);

    printf("\nc) Kernel 3: ");
    for (int i = 0; i < ARR_LEN; i++)
        printf("%d, ", c3[i]);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}