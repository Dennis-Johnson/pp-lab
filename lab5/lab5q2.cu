#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define ARR_LEN 12

/*
 * Q2. Sort an array of size ARR_LEN using parallel selection sort. 
*/

__global__ void selectionSort(int *arr, int *result, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id > n)
        return;

    int pos = 0;
    for (int i = 0; i < n; i++)
        if (arr[id] > arr[i] || (arr[id] == arr[i] && id > i))
            pos++;

    result[pos] = arr[id];
}

int main()
{
    // host copies of variables arr, result
    int arr[ARR_LEN] = {1, 7, 8, 2, 3, 6, 9, 5, 4, 12, 11, 10};
    int result[ARR_LEN];

    printf("Original Array: ");
    for (int i = 0; i < ARR_LEN; i++)
        printf("%d ", arr[i]);
    printf("\n");

    // device copies of array arr and result
    int *d_arr, *d_result;

    int size = ARR_LEN * sizeof(int);

    // Allocate space for device copy of arr and result
    cudaMalloc((void **)&d_arr, size);
    cudaMalloc((void **)&d_result, size);

    // Copy inputs to device
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // Launch selection sort kernel on GPU
    selectionSort<<<1, ARR_LEN>>>(d_arr, d_result, ARR_LEN);

    // Copy result to result array
    cudaError err = cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("Sorted Array:   ");
    for (int i = 0; i < ARR_LEN; i++)
        printf("%d ", result[i]);
    printf("\n");

    // Cleanup
    cudaFree(d_arr);
    cudaFree(d_result);

    return 0;
}