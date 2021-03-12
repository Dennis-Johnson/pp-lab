#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char  *argv[]){
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int N = size, M = 0;
	float collected[N], indv_avg = 0;
	int *Arr;

	if(rank == 0){
		printf("Enter value for M: ");
		scanf("%d", &M);
		
		Arr = calloc(N*M, sizeof(int));
	
		printf("Enter %d values\n", N*M);
		fflush(stdout);
		for(int i = 0; i < M*N; i++)
			scanf("%d", &Arr[i]);
	}

	int recv[M];
	MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(Arr, M, MPI_INT, recv, M, MPI_INT, 0, MPI_COMM_WORLD);
	fflush(stdout);

	//Compute individual average
	for(int i = 0; i < M; i++){
		indv_avg += recv[i];
	}
	indv_avg /= M;

	printf("Process %d computed indv_average %f\n", rank, indv_avg);

	MPI_Gather(&indv_avg, 1, MPI_FLOAT, collected, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	if(rank == 0){
		float global_avg = 0;
		
		fflush(stdout);
		for(int i = 0; i < N; i++){
		 	//printf("%f\t", collected[i]);
			global_avg += collected[i]; 
		}
		global_avg /= N;

		printf("Global Average is %f\n", global_avg);
		fflush(stdout);
	}

 	MPI_Finalize();
	return 0;
}
