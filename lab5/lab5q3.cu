#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <device_functions.h>

#define ARR_LEN 12
#define swap(A, B)    \
    {                 \
        int temp = A; \
        A = B;        \
        B = temp;     \
    }

/*
 * Q3. Sort an array of size ARR_LEN using parallel odd-even transposition sort. 
*/

__global__ void oddEvenSort(int *arr, int n)
{
    int len = (n + 1) / 2;

    int index = threadIdx.x;
    int isOdd = index & 1;
    int isWithinBounds = (index < (n - 1));

    for (int i = 0; i < len; i++)
    {
        // even cycle
        if (!isOdd && isWithinBounds)
        {
            if (arr[index] > arr[index + 1])
                swap(arr[index], arr[index + 1]);
        }
        __syncthreads();

        // odd cycle
        if (isOdd && isWithinBounds)
        {
            if (arr[index] > arr[index + 1])
                swap(arr[index], arr[index + 1]);
        }

        __syncthreads();
    }
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

    // device copies of array arr
    int *d_arr;

    int size = ARR_LEN * sizeof(int);

    // Allocate space for device copy of arr
    cudaMalloc((void **)&d_arr, size);

    // Copy inputs to device
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    // Launch oddEven sort kernel on GPU
    oddEvenSort<<<1, ARR_LEN>>>(d_arr, ARR_LEN);

    // Copy result to result array
    cudaError err = cudaMemcpy(&result, d_arr, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("Sorted Array:   ");
    for (int i = 0; i < ARR_LEN; i++)
        printf("%d ", result[i]);
    printf("\n");

    // Cleanup
    cudaFree(d_arr);
    return 0;
}