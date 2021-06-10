#include <stdio.h>
#include <cuda.h>

/* Lab8 Additional 1
* Input matrix MxN
* Output matrix MxN where each element is the sum of elements in the same row and col. 
*/
__global__ void rowColSum(int* a, int *b, int m, int n){
    int tid = threadIdx.x;
    
    // Get row and col index
    int rowIndex = tid % m; 
    int colIndex = tid % n; 
    int rowSum = 0, colSum = 0;
    
    // Column sum
    for(int i = 0; i < m; i++)
      colSum += a[i * n + colIndex];
    
    // Row sum
    for(int j = 0; j < n; j++)
      rowSum += a[rowIndex * n + j];

    // Subtract the element as it's been added twice
    b[tid] = rowSum + colSum;
}

int main(){
    const int m = 2;
    const int n = 5;
    int index = 0;

    // Input and output 2D matrices, flattened
    int a[m * n]; 
    int b[m * n]; 

    // Initialize A and B
    for(int i = 0; i < m*n; i++){
          a[i] = i + 1;
          b[i] = 0;
    }


    printf("A is : \n");
    for(int i = 0; i < m*n; i++)
        printf("%d ", a[i]);
    
    
    // Array size
    size_t size_arr = m * n * sizeof(int);

    // Device copies of matrices
    int *d_a, *d_b;

    // Allocate memory on device
    cudaMalloc((void**) &d_a, size_arr);
    cudaMalloc((void**) &d_b, size_arr);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size_arr, cudaMemcpyHostToDevice);

    // Launch Kernel
    const int _numthreads = m * n;
    rowColSum<<<1, _numthreads>>>(d_a, d_b, m, n);

    // Copy outputs back from device
    cudaError err = cudaMemcpy(b, d_b, size_arr, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) 
      printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("\nA is :");
    for(int i = 0; i < m*n; i++){
      if(i % n == 0)
        printf("\n");
        
      printf("%d ", a[i]);
    }

    printf("\n\nB is :");
    for(int i = 0; i < m*n; i++){
      if(i % n == 0)
        printf("\n");

      printf("%d ", b[i]);
    }

    // Free and cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}
