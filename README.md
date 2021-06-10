# Parallel Programming Lab - MIT CSE - Sem 6 

- **Lab 1 -->** MPI Point-to-Point Communication (Missed it)
- **Lab 2 -->** MPI Point-to-Point Communication
- **Lab 3 -->** MPI Collective Communication
- **Lab 4 -->** MPI Collective Communication with Error Handling
- **Lab 5 -->** CUDA programs on arrays
- **Lab 6 -->** CUDA programs on matrices
- **Lab 7 -->** CUDA programs on strings
- **Lab 8 -->** CUDA programs on matrices (continued)

# Running MPI Examples
Uses OpenMPI's wrapper compiler for C.

#### To compile
`mpicc my_app.c -o my_app`

#### Execute the serial / parallel jobs
Note: mpirun, mpiexec, and orterun are synonyms. 
Specify number of MPI processes -np

`mpirun -np 4 my_app`

# Running CUDA Examples

Everything CUDA related was done using Google Colab's free Nvidia GPUs.

Copy the required `.cu` file contents to the cell with double magic `%%cu` on top in `cuda_c_template.ipynb`
