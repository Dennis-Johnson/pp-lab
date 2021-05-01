#include <stdio.h>
#include <math.h>
#include "mpi.h"

/* Approximate the value of pi by repeated addition of the integral
 * by each process. Reduce partial areas to get the sum in the root process. 
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

int main(int argc, char *argv[]){
  int rank, size, errno;
  double pi = 0;

  MPI_Init(&argc, &argv);
  errno = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  errno = MPI_Comm_size(MPI_COMM_WORLD, &size);

  double x1 = rank * (1.0 / size);
  double x2 = x1 + (1.0 / size);
  
  double midpoint = 0.5 * (x1 + x2);
  double height = 4.0 / (1.0 + midpoint * midpoint);

  double area = (x2 - x1) * height;
  errno = MPI_Reduce(&area, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  printf("Rank %d: x1, x2, midpoint, area: %lf %lf %lf %lf\n", rank, x1, x2, midpoint, area);

  if(rank == 0)
    printf("PI = %lf\n", pi);
  
  errno = MPI_Finalize();
  return 0;
}
