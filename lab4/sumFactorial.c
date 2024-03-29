#include <stdio.h>
#include "mpi.h"

/* Using N processes, calculate the series sum of 
 * factorials upto N terms utilising MPI_Scan
 * 
 * factsum = 1! + 2! + ... + N!
 * 
 * */

void ErrorHandler(int errno){
   if (errno != MPI_SUCCESS){
     		char errstring[MPI_MAX_ERROR_STRING];
     		int errstring_len, errclass;
     		MPI_Error_class(errno, &errclass);
     		MPI_Error_string(errno, errstring, &errstring_len);
     		printf("%d %s\n",  errclass, errstring);
   }
}

int main(int argc, char *argv[]){
  int rank, size, fact=1, factsum, i, errno;

  MPI_Init(&argc, &argv);

  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  errno = MPI_Comm_rank(MPI_COMM_WORLD, &rank); ErrorHandler(errno);
  errno = MPI_Comm_size(MPI_COMM_WORLD, &size); ErrorHandler(errno);

  for(i=1; i<=rank+1; i++)
    fact = fact * i;

  errno = MPI_Scan(&fact, &factsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  ErrorHandler(errno);

  printf("Rank %d: Series sum for factorial terms = %d\n", rank, factsum);

  errno = MPI_Finalize();
  ErrorHandler(errno);

  return 0;
}
