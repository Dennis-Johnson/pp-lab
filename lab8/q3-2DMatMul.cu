#include <stdio.h>
#include <cuda.h>

/* Lab8 Q3
* Write a program in CUDA to perform matrix multiplication using 2D Grid and 2D Block.
*/
__global__ void matMul2d(const int* a, const int *b, int *c, int m, int n, int p){
    // Calculate appropriate row and col:
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row * p + col] = 0;
    
    // Compute a single element 
    for(int k = 0; k < n ; k++)
      c[row * p + col] += a[row * n + k ] * b[k * p + col];
}

int main(){
    int m = 4, n = 2, p =4;
    int a[m][n];
    int b[n][p];
    int c[m][p];

    // Initialize A and B
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j ++)
          a[i][j] = i * m + j;
    
    for(int i = 0; i < n; i++)
        for(int j = 0; j < p; j ++)
          b[i][j] = i * n + j;

    // Device copies of inputs
    int *d_a, *d_b, *d_c;

    // Allocate memory on device
    cudaMalloc((void**) &d_a, m * n * sizeof(int));
    cudaMalloc((void**) &d_b, n * p * sizeof(int));
    cudaMalloc((void**) &d_c, m * p * sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_a, a, m * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, a, n * p * sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernel on 2D grid with a 2D block
    dim3 grid((m * p / 2), (m * p / 2));
    dim3 block(2, 2);

    matMul2d<<<grid, block>>>(d_a, d_b, d_c, m, n, p);

    // Copy outputs back from device
    cudaError err = cudaMemcpy(c, d_c, m * p * sizeof(int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) 
      printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("A is : \n");
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j ++)
          printf("%d ", a[i][j]);
        printf("\n");
    }

    printf("\nB is : \n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < p; j ++)
          printf("%d ", b[i][j]);
        printf("\n");
    }

    printf("\nC is : \n");
    for(int i = 0; i < m; i++){
        for(int j = 0; j < p; j ++)
          printf("%d ", c[i][j]);
        printf("\n");
    }
           
    // Free and cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}
