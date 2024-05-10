#ifndef MAIN_H
#define MAIN_H
#include "mmio.h"
#include "common.h"

void usage(int argc, char** argv);
void print_matrix_info(char* fileName, MM_typecode matcode, 
                       int m, int n, int nnz);
void check_mm_ret(int ret);
void read_info(char* fileName, int* is_sym);
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val, 
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        double** csr_vals);
void read_vector(char* fileName, double** vector, int* vecSize);
void spmv_coo(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              double* csr_vals, int m, int n, int nnz, 
              double* vector_x, double *res, omp_lock_t* writelock);
void spmv_coo_mpi(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              double* csr_vals, int m, int n, int nnz, 
              double* vector_x, double *res, omp_lock_t* writelock);
void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res);
void spmv_coo_ser(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
                  double* csr_vals, int m, int n, int nnz, 
                  double* vector_x, double *res);
void spmv_ser(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              double* csr_vals, int m, int n, int nnz, 
              double* vector_x, double *res);
void store_result(char *fileName, double* res, int m);
void print_time(double timer[]);
void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind, 
                     double** val);
void init_locks(omp_lock_t** locks, int m);
void destroy_locks(omp_lock_t* locks, int m);
#endif

