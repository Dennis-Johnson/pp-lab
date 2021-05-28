%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define MAX_LEN 256

/* Lab 7: Programs on strings
*  Q2: Write a CUDA program that reads a string Sin and produces the string Sout as follows:
*      E.g: Input string: "PCAP" 
            Output string: "PCAPPCAPCP"
*/

__constant__ int len_str;
__constant__ int len_result;

__global__ void stringPattern(char *str, char *result){
    int indx = threadIdx.x;

    /* E.g: for str = "PCAP" --> offset(thread0) = 10 - (4 + 3 + 2 + 1) = 0   
    *                            offset(thread1) = 10 - (3 + 2 + 1)     = 4
    *                            etc...
    */
    int offset = len_result - ((len_str-indx) * (len_str-indx+1) / 2);
    int num_chars = len_str - indx;

    for(int i = 0; i < num_chars; i++)
        result[offset + i] = str[i]; 
}

int main() {

  // host copies of input and result strings
  char str[MAX_LEN] = "PCAP";
  char result[MAX_LEN] = "";
  int len_str_h = strlen(str);
  int len_result_h = len_str_h * (len_str_h + 1) / 2;

  // device copies of variables
  char *d_str, *d_result;

  size_t sizeStr = len_str_h * sizeof(char);
  size_t sizeResult = len_result_h * sizeof(char);

  // Allocate space for device copies of variables
  cudaMalloc((void **)&d_str, sizeStr);
  cudaMalloc((void **)&d_result, sizeResult);

  // Copy string lengths to constant memory on device
  cudaMemcpyToSymbol(len_str, &len_str_h, sizeof(int));
  cudaMemcpyToSymbol(len_result, &len_result_h, sizeof(int));

  // Copy inputs to device
  cudaMemcpy(d_str, str, sizeStr, cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, result, sizeResult, cudaMemcpyHostToDevice);

  // Launch kernels on GPU:
  cudaError err;

  // A thread for each starting location of repetition (equal to strlen(str))
  stringPattern<<<1, len_str_h>>>(d_str, d_result);

  // Retrieve result
  err = cudaMemcpy(result, d_result, sizeResult, cudaMemcpyDeviceToHost);

  if(err != cudaSuccess) 
    printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
  
  printf("Input string: %s\nResult: %s\n", str, result);

  // Free resources
  cudaFree(d_str);
  cudaFree(d_result);

  return 0;
}
