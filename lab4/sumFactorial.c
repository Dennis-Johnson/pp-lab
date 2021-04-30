#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[]){
  int rank, size, fact=1, factsum, i;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for(i=1; i<=rank+1; i++)
    fact = fact * i;

  MPI_Scan(&fact, &factsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  printf("Rank %d: Series sum for factorial terms = %d\n", rank, factsum);

  MPI_Finalize();
  return 0;
}
