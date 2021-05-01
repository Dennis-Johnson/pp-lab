#include <stdio.h>
#include "mpi.h"
#include <math.h>
#include <unistd.h>

void ErrorHandler(int errno){
   if (errno != MPI_SUCCESS){
     		char errstring[MPI_MAX_ERROR_STRING];
     		int errstring_len, errclass;
     		MPI_Error_class(errno, &errclass);
     		MPI_Error_string(errno, errstring, &errstring_len);
     		printf("%d %s\n",  errclass, errstring);
   }
}

int main(int argc,char* argv[])
{
	int rank, size, errno;
	int i, j, ele, mat[3][3];

	MPI_Init(&argc,&argv);
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  errno = MPI_Comm_rank(MPI_COMM_WORLD,&rank); ErrorHandler(errno);
	errno = MPI_Comm_size(MPI_COMM_WORLD,&size);

	if(rank==0)
	{	
		for(i=0; i<3; i++)
			for(j=0; j<3; j++)
				scanf("%d", &mat[i][j]);

		printf("Enter an element to search for : ");
		scanf("%d", &ele);
	}

	int temp[3];
	errno = MPI_Scatter(mat, 3, MPI_INT, temp, 3, MPI_INT, 0, MPI_COMM_WORLD);
	ErrorHandler(errno);

	printf("%d: ", rank);
	for(i=0; i<3; i++) {
		printf("%d ", temp[i]);
	}
	printf("\n");

	errno = MPI_Bcast(&ele, 1, MPI_INT, 0, MPI_COMM_WORLD);
	ErrorHandler(errno);

	int count = 0;
	for(int i=0; i<3; i++)
		if(temp[i] == ele)
			count++;

	int total;
	errno = MPI_Reduce(&count, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	ErrorHandler(errno);

	if(rank == 0)
		printf("Number of occurences of %d is %d\n",ele,total);
	
  errno = MPI_Finalize();
  ErrorHandler(errno);
  
	return 0;
}

