#include "mpi.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int isVowel(char ch){
	if(ch =='a'||ch=='e'||ch=='i'||ch=='o'||ch=='u')
		return 1;
	return 0;
}

int main(int argc, char  *argv[]){
	int rank, size, str_len = 0, indv_count = 0;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int N = size, len_each = 0;
	char str[256];
	
	if(rank == 0){
		printf("Enter contents of string\n");
		fflush(stdout);
		gets(str);
		str_len = strlen(str);
		
		if(str_len % N != 0){
			fprintf(stderr, "String length is not divisible by N\n");
			exit(1);
		}	
		
		len_each = str_len / N;	
	}
	
	char recv[len_each];	
	int collected[N];
	
	MPI_Bcast(&len_each, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(str, len_each, MPI_CHAR, recv, len_each, MPI_CHAR, 0, MPI_COMM_WORLD);
	fflush(stdout);

	printf("Process %d received %s\n", rank, recv);
	for(int i=0; i <len_each; i++){
		if(isVowel(recv[i]))
			indv_count++;
	}
	
	MPI_Gather(&indv_count, 1, MPI_INT, collected, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rank == 0){
		int total_count = 0;
		
		fflush(stdout);
		for(int i = 0; i < N; i++){
			total_count += collected[i]; 
		}

		printf("Total vowel count is %d\n", total_count);
		fflush(stdout);
	}

 	MPI_Finalize();
	return 0;
}
