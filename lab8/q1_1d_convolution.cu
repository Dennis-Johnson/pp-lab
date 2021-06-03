%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define WIDTH 8
#define MASK_WIDTH 5

/* Lab8 Q1
* Write a CUDA program to perform convolution operation on a 
* one dimensional input array N of len WIDTH 
* using a mask array M of len MASK_WIDTH 
* to produce the resultant one dimensional array P of len WIDTH
* using constant Memory for Mask array.
* Add another kernel function to the same program to perform 1D convolution using shared memory.
* Find and display the time taken by both the kernels.
*/

__constant__ int M[MASK_WIDTH];

__global__ void convolution_1D_constantMemMask(
    int *N,        // Array of length 'width'
    int *result,   // Array of length 'width'
    int width,      
    int mask_width // Mask is stored in constant memory
){
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    const int start_index = indx - (mask_width / 2);
  
    int accumulator = 0; 
    for(int j = 0; j < mask_width; j++){
        if(start_index + j >= 0 && start_index + j < width){
          accumulator += N[start_index + j] * M[j];
        }
    }
  
    result[indx] = accumulator;
}

__global__ void convolution_1D_sharedMemMask(
    int *N,        // Array of length 'width'
    int *result,   // Array of length 'width'
    int width,     // Width of array to be stored in shared memory
    int mask_width // Width of mask; mask is in constant memory
){
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    const int start_index = indx - (mask_width / 2);

    // dynamic shared memory
    extern __shared__ int shared_arr[];

    // Load into shared memory. 
    // Indexed using threadIdx.x as shared memory is within a single block.
    shared_arr[threadIdx.x] = N[indx];
    __syncthreads();
  
    int accumulator = 0; 
    for(int j = 0; j < mask_width; j++){
        if(start_index + j >= 0 && start_index + j < width){
          accumulator += N[start_index + j] * M[j];
        }
    }
  
    result[indx] = accumulator;
}

int main(){

    // Host copies of array and mask
    const int N_h[WIDTH] = {1,2,3,4,5,6,7,8};
    const int M_h[MASK_WIDTH] = {3,4,5,4,3};
    int result_h[WIDTH];

    // Size of array N and result array 
    size_t _sizeN = sizeof(int) * WIDTH;

    // Device copies
    int *d_N, *d_result;

    // Timing related stuff
    float timeElapsedA, timeElapsedB;
    cudaEvent_t beginA, beginB, endA, endB;
    cudaEventCreate(&beginA);
    cudaEventCreate(&beginB);
    cudaEventCreate(&endA);
    cudaEventCreate(&endB);

    // Copy mask to Constant Memory
    cudaMemcpyToSymbol(M, M_h, sizeof(int) * MASK_WIDTH);

    // Allocate memory on device
    cudaMalloc((void **)&d_N, _sizeN);
    cudaMalloc((void **)&d_result, _sizeN);

    // Copy inputs
    cudaMemcpy(d_N, N_h, _sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result_h, _sizeN, cudaMemcpyHostToDevice);


    cudaError err;

    // a) Launch kernel where mask is stored in constant memory and time it -------
    cudaEventRecord(beginA, 0);
    convolution_1D_constantMemMask<<<1, WIDTH>>>(d_N, d_result, WIDTH, MASK_WIDTH);
    cudaEventRecord(endA, 0);
    cudaEventSynchronize(endA);
    cudaEventElapsedTime(&timeElapsedA, beginA, endA);

    // Retrieve and print result of 1D convolution
    err = cudaMemcpy(result_h, d_result, _sizeN, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) 
      printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("Result of 1D conv: constant memory MASK:\n");
    for(int i = 0; i < WIDTH; i++)
      printf("%d ", result_h[i]);
    printf("\n\n");



    // b) Launch kernel where array N is to be stored in dynamic shared memory ------
    // A third argument is passed indicating the size of shared memory required.
    cudaEventRecord(endB, 0);
    convolution_1D_sharedMemMask<<<1, WIDTH, _sizeN>>>(d_N, d_result, WIDTH, MASK_WIDTH);
    cudaEventRecord(endB, 0);
    cudaEventSynchronize(endB);
    cudaEventElapsedTime(&timeElapsedB, beginB, endB);

    // Retrieve and print result of 1D convolution
    err = cudaMemcpy(result_h, d_result, _sizeN, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) 
      printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("Result of 1D conv: constant memory MASK & shared memory N:\n");
    for(int i = 0; i < WIDTH; i++)
      printf("%d ", result_h[i]);
    printf("\n\n");

    printf("Time elapsed for kernel a) : %f\n", timeElapsedA);
    printf("Time elapsed for kernel b) : %f\n", timeElapsedB);
     
    cudaFree(d_N);
    cudaFree(M);
    cudaFree(d_result);

    return 0;
}
