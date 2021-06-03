%%cu 
#include <stdio.h>
#include <cuda.h>

/* Lab8 Q2
* Write a program in CUDA to perform parallel Sparse Matrix Vector Multiplication 
* using compressed sparse row (CSR) storage format. 
* Represent the input sparse matrix in CSR format in the host code. */
__global__ void sparseMatMulCSR(
    int num_rows, 
    int *data, 
    int *col_index, 
    int *row_ptr,
    int *x,
    int *y
){  
  int row = threadIdx.x;
  if(row < num_rows){
      int accumulator = 0;
      int row_start = row_ptr[row];
      int row_end = row_ptr[row + 1];

      for(int i = row_start; i < row_end; i++)
        accumulator += data[i] * x[col_index[i]];

      y[row] = accumulator;
  }
}

int main(){
    const int n = 4;
    int y[n], row_ptr[n+1];
    int ipmat[n][n] = {{3,0,1,0},{0,0,0,0},{0,2,4,1},{1,0,0,1}};
    int x[] = {3,4,5,6};
    int nonzerocount = 0;
    /* Find the num of nonzero elements in input matrix
    *  Populate the row_ptr array
    */
    printf("Input matrix is \n");
    for(int i = 0; i < n; i++){
        row_ptr[i] = nonzerocount;
        for(int j = 0; j < n; j ++){
            if(ipmat[i][j] != 0)
              nonzerocount++;

            printf("%d ", ipmat[i][j]);
        }
        printf("\n");
    }
    // Last entry is total num of nonzero elements
    row_ptr[n] = nonzerocount;

    int data[nonzerocount], col_index[nonzerocount];
    int k = 0;

    /* Populate the col_index array
    *  Populte the data array
    */
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(ipmat[i][j] != 0){
                data[k] = ipmat[i][j];
                col_index[k++] = j;
            }
        }
    }
    
    printf("\nData array\n");
    for(int i = 0; i < nonzerocount; i++)
      printf("%d ", data[i]);

    printf("\nCol_Index array\n");
    for(int i = 0; i < nonzerocount; i++)
      printf("%d ", col_index[i]);

    printf("\nRow_Ptr array\n");
    for(int i = 0; i <= n; i++)
      printf("%d ", row_ptr[i]);

    printf("\nInput vector X to multiply\n");
    for(int i = 0; i < n; i++)
      printf("%d ", x[i]);

    // Device copies of inputs
    int *d_data, *d_col_index, *d_row_ptr, *d_x, *d_y;

    // Allocate memory on device
    cudaMalloc((void**) &d_data, nonzerocount * sizeof(int));
    cudaMalloc((void**) &d_col_index, nonzerocount * sizeof(int));
    cudaMalloc((void**) &d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc((void**) &d_x, n * sizeof(int));
    cudaMalloc((void**) &d_y, n * sizeof(int));

    // Copy inputs to device
    cudaMemcpy(d_data, data, nonzerocount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index, col_index, nonzerocount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, row_ptr, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernel
    sparseMatMulCSR<<<1, n>>>(n, d_data, d_col_index, d_row_ptr, d_x, d_y);

    // Copy outputs back from device
    cudaError err = cudaMemcpy(y, d_y, n * sizeof(int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) 
      printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));

    printf("\nThe transformed input vector is: \n");
    for(int i = 0; i < n; i++)
      printf("%d ", y[i]);

    // Free and cleanup
    cudaFree(d_data);
    cudaFree(d_col_index);
    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
