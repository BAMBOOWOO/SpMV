This project involves implementing parallel sparse matrix-vector multiply (SpMV) on CPUs using OpenMP and OpenMPI.

1)Please read Section 3 of https://www.nvidia.com/docs/IO/77944/sc09-spmv-throughput.pdf
Understand how COO and CSR formats work.

2)  Test the code on the 'cant' matrix and make sure the answer matches the provided anwer. 
  a) Use 'diff' against cant/ans.mtx
  b) Example usage is: ./spmv cant/cant.mtx cant/b.mtx ./myans.mtx 
     Then: diff ./myans.mtx cant/ans.mtx
    

