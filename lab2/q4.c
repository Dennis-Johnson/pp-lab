#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]){
	int rank, size, num, recv_num;
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 0){
		printf("Enter a number to send out: ");
		scanf("%d", &num);

		MPI_Send(&num, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
		MPI_Recv(&recv_num, 1, MPI_INT, size-1, 1, MPI_COMM_WORLD, &status);

		printf("Root --> Received %d from Rank %d.\n", recv_num, size - 1);
	}
	else {
		MPI_Recv(&num, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &status);
		printf("Rank %d --> Received %d from Rank %d\n", rank, num, rank - 1);

		num += 1;
		if(rank == size - 1)
			MPI_Send(&num, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
		else
			MPI_Send(&num, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}