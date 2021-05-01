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
	int rank,size,errno;
	MPI_Init(&argc,&argv);
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	int a[3][3];
	int ele;
	errno = MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	errno = MPI_Comm_size(MPI_COMM_WORLD,&size);
	if(rank==0)
	{	
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				scanf("%d",&a[i][j]);
			}
		}

		printf("enter an element to be searched: ");
		scanf("%d",&ele);
	}
	int b[3];
	errno=MPI_Scatter(a,3,MPI_INT,b,3,MPI_INT,0,MPI_COMM_WORLD);
	ErrorHandler(errno);

	printf("%d=>", rank);
	for(int i=0;i<3;i++) {
		printf("%d ", b[i]);
	}
	printf("\n");
	errno=MPI_Bcast(&ele,1,MPI_INT,0,MPI_COMM_WORLD);
	ErrorHandler(errno);
	int count=0;
	for(int i=0;i<3;i++)
	{
		if(b[i]==ele)
		{
			count++;
		}
	}
	int total;
	errno=MPI_Reduce(&count,&total,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	ErrorHandler(errno);
	if(rank==0)
	{
		printf("Number of occurences of %d is %d\n",ele,total);
	}
	MPI_Finalize();
	return 0;
}

