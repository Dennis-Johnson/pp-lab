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

int main(){

    // Host copies of array and mask
    const int N_h[WIDTH] = {1,2,3,4,5,6,7,8};
    const int M_h[MASK_WIDTH] = {3,4,5,4,3};
    int result_h[WIDTH];

    // Size of array N and result array 
    size_t _sizeN = sizeof(int) * WIDTH;

    // Device copies
    int *d_N, *d_result;

    // Copy mask to Constant Memory
    cudaMemcpyToSymbol(M, M_h, sizeof(int) * MASK_WIDTH);

    // Allocate memory on device
    cudaMalloc((void **)&d_N, _sizeN);
    cudaMalloc((void **)&d_result, _sizeN);

    // Copy inputs
    cudaMemcpy(d_N, N_h, _sizeN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result_h, _sizeN, cudaMemcpyHostToDevice);

    cudaError err;

    // a) Launch kernel where mask is stored in constant memory -------------------
    convolution_1D_constantMemMask<<<1, WIDTH>>>(d_N, d_result, WIDTH, MASK_WIDTH);

    // Retrieve and print result of 1D convolution
    err = cudaMemcpy(result_h, d_result, _sizeN, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) 
      printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    for(int i = 0; i < WIDTH; i++)
      printf("%d ", result_h[i]);
    printf("\n");

     // b) Launch kernel where mask is stored in shared memory ------------------
    //convolution_1D_constantMemMask<<<1, WIDTH>>>(d_N, d_result, WIDTH, MASK_WIDTH);

    // Retrieve and print result of 1D convolution
    err = cudaMemcpy(result_h, d_result, _sizeN, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) 
      printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    for(int i = 0; i < WIDTH; i++)
      printf("%d ", result_h[i]);
    printf("\n");
     
    cudaFree(d_N);
    cudaFree(M);
    cudaFree(d_result);

    return 0;
}
