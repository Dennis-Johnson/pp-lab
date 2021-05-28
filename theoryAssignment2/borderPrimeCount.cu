#include <stdio.h>
#include <cuda.h>

/* Read a square matrix (N*N) and count 
*  the no. of prime numbers on its border.
*  Use 2,2 grid and a 2D block.  
*/

__device__ bool checkPrime(int num){
    bool isPrime = 1;

    // Check if num is a prime number
    for(int i = 2; i <= sqrtf(num); i++){
      if(num % i == 0){
        isPrime = 0;
        break;
      }
    } 

    if (num <= 1) 
      isPrime = 0;

    return isPrime;
}

// Each thread determines if it's on a prime valued border element 
// If it is, it increments primeCount atomically
__global__ void borderPrimeCount(int *mat, int *primeCount, int N){
    int _blockId = (gridDim.x * blockIdx.y ) + blockIdx.x;

    int globalTid = (_blockId * blockDim.x * blockDim.y)
                  + (threadIdx.y * blockDim.x)
                  +  threadIdx.x;

    int ele = mat[globalTid];

    /* Condition 1 --> Top or Bottom border 
    *  Condition 2 --> Left border
    *  Condition 3 --> Right border 
    */
    if(globalTid < N || globalTid >= N * (N-1)
    || globalTid % N == 0                      
    || globalTid % N == (N-1)){
        if(checkPrime(ele))
          atomicAdd(primeCount, 1);
    }
}

int main()
{
    const int N = 6;
 
    // host copy of matrix
    int mat[N * N] = {
                    0, 1, 2, 3, 4, 5,
                   1, 2, 3, 4, 5, 6,
                   2, 3, 4, 5, 6, 7, 
                   3, 4, 5, 6, 7, 8,
                   4, 5, 6, 7, 8, 9,
                   5, 6, 7, 8, 9, 10
                   };
 
    int primeCount = 0;
 
    // device copies of variables
    int *d_mat, *d_primeCount;
    int sizeMat = N * N * sizeof(int);

    // Allocate space for device copies of a, b, sum
    cudaMalloc((void **)&d_mat, sizeMat);
    cudaMalloc((void **)&d_primeCount, sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_mat, mat, sizeMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primeCount, &primeCount, sizeof(int), cudaMemcpyHostToDevice);

    cudaError err;
    
    // Use a (2,2) grid and 2D block:
    dim3 dimGrid(2,2,1);
    dim3 dimBlock(N/2, N/2, 1);
 
    // Launch kernel on GPU
    borderPrimeCount<<<dimGrid, dimBlock>>>(d_mat, d_primeCount, N);
    err = cudaMemcpy(&primeCount, d_primeCount, sizeof(int), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
        printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("Found %d primes on border elements\n", primeCount);
 
    // Cleanup
    cudaFree(d_mat);
    cudaFree(d_primeCount);
    return 0;
}
