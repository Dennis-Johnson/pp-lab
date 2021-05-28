#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define MAX_LEN 256

/* Lab 7: Programs on strings
*  Q1: Write a CUDA program to count the number of times
*     a given word is repeated in a sentence. (Use atomic function)
*/

__constant__ int len_sentence;
__constant__ int len_word;

__global__ void countWord(char *sentence, char *word, int *occurences){
    int indx = threadIdx.x;

    int flag = 0;
    for(int i = 0; i < len_word; i++){
        if(word[i] != sentence[i + indx]){
            flag = 1;
            break;
        }
    }

    if(flag == 0)
      atomicAdd(occurences, 1);
}

int main() {

  // host copies of sentence and word strings
  char sentence[MAX_LEN] = "This is a input sentence string string input string";
  char word[MAX_LEN] = "string";
  int len_sentence_h = strlen(sentence);
  int len_word_h = strlen(word);

  // number of occurences of 'word' in 'string'
  int occurences = 0;
 
  // device copies of variables
  char *d_sentence, *d_word;
  int  *d_occurences;

  size_t sizeSentence = len_sentence_h * sizeof(char);
  size_t sizeWord = len_word_h * sizeof(char);

  // Allocate space for device copies of variables
  cudaMalloc((void **)&d_sentence, sizeSentence);
  cudaMalloc((void **)&d_word, sizeWord);
  cudaMalloc((void **)&d_occurences, sizeof(int));

  // Copy string lengths to constant memory on device
  cudaMemcpyToSymbol(len_sentence, &len_sentence_h, sizeof(int));
  cudaMemcpyToSymbol(len_word, &len_word_h, sizeof(int));

  // Copy inputs to device
  cudaMemcpy(d_sentence, sentence, sizeSentence, cudaMemcpyHostToDevice);
  cudaMemcpy(d_word, word, sizeWord, cudaMemcpyHostToDevice);
  cudaMemcpy(d_occurences, &occurences, sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernels on GPU:
  cudaError err;

  // A thread for each possible starting index of 'word'
  int num_threads = len_sentence_h - len_word_h + 1;
  countWord<<<1, num_threads>>>(d_sentence, d_word, d_occurences);

  // Retrieve result
  err = cudaMemcpy(&occurences, d_occurences, sizeof(int), cudaMemcpyDeviceToHost);

  if(err != cudaSuccess) 
    printf("CUDA error copying to Host: %s\n", cudaGetErrorString(err));
  
  printf("Found %s in sentence %d times\n", word, occurences);

  // Free resources
  cudaFree(d_sentence);
  cudaFree(d_word);
  cudaFree(d_occurences);

  return 0;
}
