#include "mpi.h"
#include <stdio.h>

int factorial(int n){
	if(n <= 2)
		return n;
	
	return n * factorial(n-1);
}

int main(int argc, char  *argv[]){
	int rank, size, N, A[10], B[10], c, i;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if(rank == 0){
		N = size;
		printf("Enter %d values\n", N);
		fflush(stdout);
		for(i = 0; i < N; i++)
			scanf("%d", &A[i]);
	}

	MPI_Scatter(A, 1, MPI_INT, &c, 1, MPI_INT, 0, MPI_COMM_WORLD);
	printf("Process %d received %d\n", rank, c);
	fflush(stdout);

	c = factorial(c);
	printf("Process %d computed %d\n", rank, c);

	MPI_Gather(&c, 1, MPI_INT, B, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(rank == 0){
		printf("Result gathered in root\n");
		fflush(stdout);
		for(i = 0; i < N; i++)
			printf("%d\t", B[i]);
		
		printf("\n");
		fflush(stdout);
	}

 	MPI_Finalize();
	return 0;
}
