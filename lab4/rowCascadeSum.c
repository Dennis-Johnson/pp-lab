#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * If given a 4x4 matrix as input, 
 * say, 1 2 3 4
 *      1 2 3 1
 *      1 1 1 1
 *      2 1 2 1
 *
 *     transform by cumulatively summing rows to get
 *     1 2 3 4
 *     2 4 6 5 
 *     3 5 7 6
 *     5 6 9 7
 */

void ErrorHandler(int errno){
   if (errno != MPI_SUCCESS){
     		char errstring[MPI_MAX_ERROR_STRING];
     		int errstring_len, errclass;
     		MPI_Error_class(errno, &errclass);
     		MPI_Error_string(errno, errstring, &errstring_len);
     		printf("%d %s\n",  errclass, errstring);
   }
}

int main (int argc, char* argv[]) {
	int rank, size, errno;
	int i, j;
	int mat[16], temp[4], sum[4];


	MPI_Init(&argc, &argv);
	errno = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	errno = MPI_Comm_size(MPI_COMM_WORLD, &size);
	
  if (rank == 0) {
	  printf("Enter the elements\n");

	  for (i = 0; i < 16; i++) 
      scanf("%d", &mat[i]);

		printf("\n");
	}

	errno = MPI_Scatter(mat, 4, MPI_INT, temp, 4, MPI_INT, 0, MPI_COMM_WORLD);
	ErrorHandler(errno);
	errno = MPI_Scan(temp, sum, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	ErrorHandler(errno);

	for (i = 0; i < 4; i++) 
		printf("%d ", sum[i]);
  printf("\n");

	errno = MPI_Finalize();
  ErrorHandler(errno);
	return 0;
}

