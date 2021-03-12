#include "mpi.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char  *argv[]){
	int rank, size, str_len = 0, indv_count = 0;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int N = size, len_each = 0;
	char str1[256], str2[256];
	
	if(rank == 0){
		printf("Enter contents of string1\n");
		fflush(stdout);
		gets(str1);

		printf("Enter contents of string1\n");
		gets(str2);
		fflush(stdout);

		str_len = strlen(str1);

		if(str_len % N != 0 || strlen(str2) != str_len){
			fprintf(stderr, "String length is not divisible by N\n");
			exit(1);
		}	
		
		len_each = str_len / N;	
	}
	
	char *indv_cat;
	char collected[256];
	
	MPI_Bcast(&len_each, 1, MPI_INT, 0, MPI_COMM_WORLD);

	char recv1[len_each], recv2[len_each];	
	strcpy("", recv1);
	strcpy("", recv2);

	MPI_Scatter(str1, len_each, MPI_CHAR, recv1, len_each, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Scatter(str2, len_each, MPI_CHAR, recv2, len_each, MPI_CHAR, 0, MPI_COMM_WORLD);
	fflush(stdout);

	printf("Process %d received %s and %s\n", rank, recv1, recv2);
	indv_cat = strcat(recv1, recv2);
	printf("Process %d concat %s\n", rank, indv_cat);
	
	MPI_Gather(indv_cat, strlen(indv_cat)+1, MPI_CHAR, collected, len_each*2, MPI_CHAR, 0, MPI_COMM_WORLD);
	
	if(rank == 0){
		char final[256];
		fflush(stdout);
		printf("Recevied at root %s\n", collected); 
	}

 	MPI_Finalize();
	return 0;
}
