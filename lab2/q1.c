#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main(int argc, char *argv[]){
	int rank, size;
	char word[20];
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 0){
		printf("Enter a word to send out: ");
		gets(word);

		int len = strlen(word);
		MPI_Ssend(&len, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
		MPI_Ssend(word, len, MPI_CHAR, 1, 1, MPI_COMM_WORLD);

		MPI_Recv(word, len, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);
		printf("Received %s back from P1\n", word);
	}
	else {
		int recv_len;
		MPI_Recv(&recv_len, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		MPI_Recv(word, recv_len, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);

		printf("Received %s from P0\n", word);

		int toggle = 0;
		for(int i = 0; i < recv_len; i++){
			if(isupper(word[i]))
				word[i] = tolower(word[i]);
			else 
				word[i] = toupper(word[i]);
		}

		MPI_Ssend(word, recv_len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}