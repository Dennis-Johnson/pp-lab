#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]){
	int rank, size, num;
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 0){
		int arr_len = size - 1;
		int arr[arr_len];

		printf("Enter %d elements into the array: ", arr_len);
		
		for(int i = 0; i < arr_len; i++)
			scanf("%d", &arr[i]);

		// int buff_size = arr_len * sizeof(int);
		int buff_size = 100;
		char *buff = malloc(buff_size);

		MPI_Buffer_attach(buff, buff_size);
		for(int i = 1; i < size; i++){
			MPI_Bsend(&arr[i-1], 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		}
		MPI_Buffer_detach(&buff, &buff_size);
	}
	else {
		MPI_Recv(&num, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

		if(rank % 2 == 0)
			printf("Rank %d --> Received %d from Root, Square = %d\n", rank, num, num*num);
		else
			printf("Rank %d --> Received %d from Root, Cube = %d\n", rank, num, num*num*num);
	}

	MPI_Finalize();
	return 0;
}