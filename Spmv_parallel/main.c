#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include "main.h"
#include <mpi.h>
#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100
#define MAX_ITER 100

#define NUM_TIMERS       9
#define LOAD_TIME        0
#define CONVERT_TIME     1
#define LOCK_INIT_TIME   2
#define SPMV_COO_TIME    3
#define SPMV_COO_MPI_TIME 4
#define SPMV_CSR_TIME    5
#define SPMV_COO_SER     6
#define SPMV_CSR_SER     7
#define STORE_TIME       8


int main(int argc, char** argv)
{
    // program info
    usage(argc, argv);

    // Initialize timess
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();


    // Read the sparse matrix file name
    char matrixName[MAX_FILENAME];
    strcpy(matrixName, argv[1]);
    int is_symmetric = 0;
    read_info(matrixName, &is_symmetric);


    // Read the sparse matrix and store it in row_ind, col_ind, and val,
    // also known as co-ordinate format (COO).
    int ret;
    MM_typecode matcode;
    int m;
    int n;
    int nnz;
    int *row_ind;
    int *col_ind;
    double *val;


    // load and expand sparse matrix from file (if symmetric)
    fprintf(stdout, "Matrix file name: %s ... ", matrixName);
    t0 = ReadTSC();
    ret = mm_read_mtx_crd(matrixName, &m, &n, &nnz, &row_ind, &col_ind, &val, 
                          &matcode);
    check_mm_ret(ret);
    if(is_symmetric) {
        expand_symmetry(m, n, &nnz, &row_ind, &col_ind, &val);
    }
    timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);

    
    // Convert co-ordinate format to CSR format
    fprintf(stdout, "Converting COO to CSR...");
    unsigned int* csr_row_ptr = NULL; 
    unsigned int* csr_col_ind = NULL;  
    double* csr_vals = NULL; 
    t0 = ReadTSC();
    // IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED
    convert_coo_to_csr(row_ind, col_ind, val, m, n, nnz,
                       &csr_row_ptr, &csr_col_ind, &csr_vals);
    timer[CONVERT_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");


    // Load the input vector file
    char vectorName[MAX_FILENAME];
    strcpy(vectorName, argv[2]);
    fprintf(stdout, "Vector file name: %s ... ", vectorName);
    double* vector_x;
    unsigned int vector_size;
    t0 = ReadTSC();
    read_vector(vectorName, &vector_x, &vector_size);
    timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);
    assert(n == vector_size);
    fprintf(stdout, "file loaded\n");


    // Calculate COO SpMV
    // first set up some locks
    t0 = ReadTSC();
    omp_lock_t* writelock; 
    init_locks(&writelock, m);
    timer[LOCK_INIT_TIME] += ElapsedTime(ReadTSC() - t0);

    double *res_coo = (double*) malloc(sizeof(double) * m);;
    assert(res_coo);
    fprintf(stdout, "Calculating COO SpMV ... ");
    t0 = ReadTSC();
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        // IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED 
        spmv_coo(row_ind, col_ind, val, m, n, nnz, vector_x, res_coo, 
                 writelock);
    }
    timer[SPMV_COO_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");

   
    

    // Calculate CSR SpMV
    double *res_csr = (double*) malloc(sizeof(double) * m);;
    assert(res_csr);
    fprintf(stdout, "Calculating CSR SpMV ... ");
    t0 = ReadTSC();
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        // IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED 
        spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, vector_x, res_csr);
    }
    timer[SPMV_CSR_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");



	// Calculate COO SpMV in serial
    double *res_coo_ser = (double*) malloc(sizeof(double) * m);;
    assert(res_coo_ser);
    fprintf(stdout, "Calculating COO SpMV serial ... ");
    t0 = ReadTSC();
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        // IMPLEMENT THIS FUNCTION 
        spmv_coo_ser(row_ind, col_ind, val, m, n, nnz, vector_x, res_coo_ser);
    }
    timer[SPMV_COO_SER] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");



    // Calculate CSR SpMV in serial
    double *res_csr_ser = (double*) malloc(sizeof(double) * m);;
    assert(res_csr_ser);
    fprintf(stdout, "Calculating CSR SpMV ... ");
    t0 = ReadTSC();
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        // IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED 
        spmv_ser(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, vector_x, 
                 res_csr_ser);
    }
    timer[SPMV_CSR_SER] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");



    // Store the calculated vector in a file, one element per line.
    char resName[MAX_FILENAME];
    strcpy(resName, argv[3]); 
    fprintf(stdout, "Result file name: %s ... ", resName);
    t0 = ReadTSC();
    store_result(resName, res_coo, m);
    // store_result(resName, res_coo, m);
    timer[STORE_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "file saved\n");


    // print timer
    print_time(timer);


    // Free memory
    free(csr_row_ptr);
    free(csr_col_ind);
    free(csr_vals);
    free(vector_x);
    free(res_coo);
    free(res_csr);
    free(res_coo_ser);
    free(res_csr_ser);
    free(row_ind);
    free(col_ind);
    free(val);
    
    destroy_locks(writelock, m);
    return 0;
}


/* This function checks the number of input parameters to the program to make 
   sure it is correct. If the number of input parameters is incorrect, it 
   prints out a message on how to properly use the program.
   input parameters:
       int    argc
       char** argv 
   return parameters:
       none
 */
void usage(int argc, char** argv)
{
    if(argc < 4) {
        fprintf(stderr, "usage: %s <matrix> <vector> <result>\n", argv[0]);
        exit(EXIT_FAILURE);
    } 
}

/* This function prints out information about a sparse matrix
   input parameters:
       char*       fileName    name of the sparse matrix file
       MM_typecode matcode     matrix information
       int         m           # of rows
       int         n           # of columns
       int         nnz         # of non-zeros
   return paramters:
       none
 */
void print_matrix_info(char* fileName, MM_typecode matcode, 
                       int m, int n, int nnz)
{
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Matrix name:     %s\n", fileName);
    fprintf(stdout, "Matrix size:     %d x %d => %d\n", m, n, nnz);
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is matrix:       %d\n", mm_is_matrix(matcode));
    fprintf(stdout, "Is sparse:       %d\n", mm_is_sparse(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is complex:      %d\n", mm_is_complex(matcode));
    fprintf(stdout, "Is real:         %d\n", mm_is_real(matcode));
    fprintf(stdout, "Is integer:      %d\n", mm_is_integer(matcode));
    fprintf(stdout, "Is pattern only: %d\n", mm_is_pattern(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is general:      %d\n", mm_is_general(matcode));
    fprintf(stdout, "Is symmetric:    %d\n", mm_is_symmetric(matcode));
    fprintf(stdout, "Is skewed:       %d\n", mm_is_skew(matcode));
    fprintf(stdout, "Is hermitian:    %d\n", mm_is_hermitian(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");

}


/* This function checks the return value from the matrix read function, 
   mm_read_mtx_crd(), and provides descriptive information.
   input parameters:
       int ret    return value from the mm_read_mtx_crd() function
   return paramters:
       none
 */
void check_mm_ret(int ret)
{
    switch(ret)
    {
        case MM_COULD_NOT_READ_FILE:
            fprintf(stderr, "Error reading file.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_PREMATURE_EOF:
            fprintf(stderr, "Premature EOF (not enough values in a line).\n");
            exit(EXIT_FAILURE);
            break;
        case MM_NOT_MTX:
            fprintf(stderr, "Not Matrix Market format.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_NO_HEADER:
            fprintf(stderr, "No header information.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_UNSUPPORTED_TYPE:
            fprintf(stderr, "Unsupported type (not a matrix).\n");
            exit(EXIT_FAILURE);
            break;
        case MM_LINE_TOO_LONG:
            fprintf(stderr, "Too many values in a line.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_COULD_NOT_WRITE_FILE:
            fprintf(stderr, "Error writing to a file.\n");
            exit(EXIT_FAILURE);
            break;
        case 0:
            fprintf(stdout, "file loaded.\n");
            break;
        default:
            fprintf(stdout, "Error - should not be here.\n");
            exit(EXIT_FAILURE);
            break;

    }
}

/* This function reads information about a sparse matrix using the 
   mm_read_banner() function and printsout information using the
   print_matrix_info() function.
   input parameters:
       char*       fileName    name of the sparse matrix file
   return paramters:
       none
 */
void read_info(char* fileName, int* is_sym)
{
    FILE* fp;
    MM_typecode matcode;
    int m;
    int n;
    int nnz;

    if((fp = fopen(fileName, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if(mm_read_banner(fp, &matcode) != 0)
    {
        fprintf(stderr, "Error processing Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    } 

    if(mm_read_mtx_crd_size(fp, &m, &n, &nnz) != 0) {
        fprintf(stderr, "Error reading size.\n");
        exit(EXIT_FAILURE);
    }

    print_matrix_info(fileName, matcode, m, n, nnz);
    *is_sym = mm_is_symmetric(matcode);

    fclose(fp);
}

/* This function converts a sparse matrix stored in COO format to CSR format.
   input parameters:
       int*	row_ind		list or row indices (per non-zero)
       int*	col_ind		list or col indices (per non-zero)
       double*	val		list or values  (per non-zero)
       int	m		# of rows
       int	n		# of columns
       int	n		# of non-zeros
   output parameters:
       unsigned int** 	csr_row_ptr	pointer to row pointers (per row)
       unsigned int** 	csr_col_ind	pointer to column indices (per non-zero)
       double** 	csr_vals	pointer to values (per non-zero)
   return paramters:
       none
 */
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val, 
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        double** csr_vals)

{
    *csr_col_ind = (unsigned int*) malloc(sizeof(unsigned int) * nnz);
    *csr_row_ptr = (unsigned int*) malloc(sizeof(unsigned int) * (m+1));
    *csr_vals = (double*) malloc(sizeof(double) * nnz);

    unsigned int *csr_col = (unsigned int*) malloc(sizeof(unsigned int) * nnz);
    unsigned int *csr_row = (unsigned int*) malloc(sizeof(unsigned int) * (m+1));
    double *csrv = (double*) malloc(sizeof(double) * nnz);
    unsigned int *temp = (unsigned int*) malloc(sizeof(unsigned int) * (m+1));
    int count = 0;
      
    
    #pragma omp parallel 
    {
        int val = 0, last_row = 0, temp_val = 0; 
        
        #pragma omp for nowait
        for(int i=0; i<nnz; i++){
            int row = row_ind[i]-1;
            temp_val++;
            if(row == last_row){
                val += temp_val;
                temp_val = 0;
            } 
            else{
                #pragma omp atomic
                csr_row[last_row+1] += val;
                last_row = row;
                val = temp_val;
                temp_val = 0;
            }
        }
        #pragma omp atomic
        csr_row[last_row+1] += val;
    } 
    
    //csr_row[0] = 0;
    for(int i=0; i<m; i++){
        csr_row[i+1] += csr_row[i]; 
    }
    
    memcpy(temp, csr_row, sizeof(unsigned int) * (m+1));

    
    for(int n=0; n<nnz; n++){
        int row  = row_ind[n] - 1;
        int dest = temp[row];
        csr_col[dest] = col_ind[n];
        csrv[dest] = val[n];
        temp[row]++;
    }
    
    //printf("csr_val: %f\n", csrv[4007380]);
    memcpy(*csr_col_ind, csr_col, sizeof(unsigned int) * (nnz));
    
    memcpy(*csr_vals, csrv, sizeof(double) * (nnz));
    
    memcpy(*csr_row_ptr, csr_row, sizeof(unsigned int) * (m+1));
    
    
}

/* Reads in a vector from file.
   input parameters:
       char*	fileName	name of the file containing the vector
   output parameters:
       double**	vector		pointer to the vector
       int*	vecSize 	pointer to # elements in the vector
   return parameters:
       none
 */
void read_vector(char* fileName, double** vector, int* vecSize)
{
    FILE* fp = fopen(fileName, "r");
    assert(fp);
    char line[MAX_NUM_LENGTH];    
    fgets(line, MAX_NUM_LENGTH, fp);
    fclose(fp);

    unsigned int vector_size = atoi(line);
    double* vector_ = (double*) malloc(sizeof(double) * vector_size);

    fp = fopen(fileName, "r");
    assert(fp); 
    // first read the first line to get the # elements
    fgets(line, MAX_NUM_LENGTH, fp);

    unsigned int index = 0;
    while(fgets(line, MAX_NUM_LENGTH, fp) != NULL) {
        vector_[index] = atof(line); 
        index++;
    }

    fclose(fp);
    assert(index == vector_size);

    *vector = vector_;
    *vecSize = vector_size;
}

/* SpMV function for COO stored sparse matrix
 */
void spmv_coo(unsigned int* row_ind, unsigned int* col_ind, double* vals, 
              int m, int n, int nnz, double* vector_x, double *res, 
              omp_lock_t* writelock)
{
    double *resvals = (double*) malloc(sizeof(double) * m);
    //printf("%d", nnz);
    double *temp;
    #pragma omp parallel shared(temp)
    {
        double cooval = 0.0, vecval = 0.0, prod = 0.0, sum = 0.0;
        int coo_c = 0, coo_r = 0, lastrow = 0;
        /*int nid = omp_get_num_threads();
        
        #pragma omp single
        {
            temp = (double*) malloc(sizeof(double) * m * nid);
        }
        
        
        int id = omp_get_thread_num();*/
        #pragma omp for
        for(int index=0; index<nnz; index++){
            
            coo_r = row_ind[index]-1;
            coo_c = col_ind[index];
            cooval = vals[index];    
            vecval = vector_x[coo_c-1];
            prod = cooval * vecval;
            if(coo_r == lastrow) sum += prod;
            else{
                #pragma omp atomic
                resvals[lastrow] += sum;
                lastrow = coo_r;
                sum = prod;
            }
            //temp[(m*id)+coo_r] += prod;
        }
        
        #pragma omp atomic
        resvals[lastrow] += sum;
        /*#pragma omp for
        for(int r=0; r<m; r++){
            double sum = 0;
            for(int j=0; j<nid; j++){
                sum += temp[(m*j)+r];
            }
            resvals[r] = sum;
        }*/
        
    }
    
    
    memcpy(res, resvals, sizeof(double) * m);
}



/* SpMV function for CSR stored sparse matrix
 */
void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res)
{
    
    double *resvals = (double*) malloc(sizeof(double) * m);
    #pragma omp parallel for schedule(static)
    for(int i=0; i<m; i++){
        double csrval = 0.0, vecval = 0.0;
        int csr_c = 0, csr_r = 0, start, end;
        double sum = 0.0;
        start = csr_row_ptr[i];
        end = csr_row_ptr[i+1];
        
        for(int index=start; index<end; index++){ 
            csr_c = csr_col_ind[index];
            csrval = csr_vals[index];
            vecval = vector_x[csr_c-1];
            sum += csrval * vecval;
        }
        resvals[i] = sum;
    }
    memcpy(res, resvals, sizeof(double) * m);
}


/* SpMV function for COO stored sparse matrix
 */
void spmv_coo_ser(unsigned int* row_ind, unsigned int* col_ind, double* vals, 
                  int m, int n, int nnz, double* vector_x, double *res)
{
    double cooval = 0.0, vecval = 0.0, prod = 0.0;
    int coo_c = 0, coo_r = 0;
    double *resvals = (double*) malloc(sizeof(double) * m);
    for(int index=0; index<nnz; index++){
        coo_r = row_ind[index];
        coo_c = col_ind[index];
        cooval = vals[index];
        
        vecval = vector_x[coo_c-1];
        prod = cooval * vecval;
        resvals[coo_r-1] += prod;
    }
    
    memcpy(res, resvals, sizeof(double) * m);
    //printf("val: %f\n", res[31097]);
}



/* SpMV function for CSR stored sparse matrix
 */
void spmv_ser(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              double* csr_vals, int m, int n, int nnz, 
              double* vector_x, double *res)
{
    double csrval = 0.0, vecval = 0.0;
    int csr_c = 0, csr_r = 0, index, start, end;
    double *resvals = (double*) malloc(sizeof(double) * m);
    for(int i=0; i<m; i++){
        double sum = 0.0;
        start = csr_row_ptr[i];
        end = csr_row_ptr[i+1];
        for(index=start; index<end; index++){ 
            csr_c = csr_col_ind[index];
            csrval = csr_vals[index];
            vecval = vector_x[csr_c-1];
            sum += csrval * vecval;
        }
        resvals[i] = sum;
    }
    memcpy(res, resvals, sizeof(double) * m);
    //printf("csrtest: %lf\n", csr_vals[0]);
}



/* Save result vector in a file
 */
void store_result(char *fileName, double* res, int m)
{
    FILE* fp = fopen(fileName, "w");
    assert(fp);

    fprintf(fp, "%d\n", m);
    for(int i = 0; i < m; i++) {
        fprintf(fp, "%0.10f\n", res[i]);
    }

    fclose(fp);
}

/* Print timing information 
 */
void print_time(double timer[])
{
    fprintf(stdout, "Module\t\tTime\n");
    fprintf(stdout, "Load\t\t");
    fprintf(stdout, "%f\n", timer[LOAD_TIME]);
    fprintf(stdout, "Convert\t\t");
    fprintf(stdout, "%f\n", timer[CONVERT_TIME]);
    fprintf(stdout, "Lock Init\t");
    fprintf(stdout, "%f\n", timer[LOCK_INIT_TIME]);
    fprintf(stdout, "COO SpMV\t");
    fprintf(stdout, "%f\n", timer[SPMV_COO_TIME]);
    fprintf(stdout, "CSR SpMV\t");
    fprintf(stdout, "%f\n", timer[SPMV_CSR_TIME]);
    fprintf(stdout, "COO SpMV Serial\t");
    fprintf(stdout, "%f\n", timer[SPMV_COO_SER]);
    fprintf(stdout, "CSR SpMV Serial\t");
    fprintf(stdout, "%f\n", timer[SPMV_CSR_SER]);
    fprintf(stdout, "Store\t\t");
    fprintf(stdout, "%f\n", timer[STORE_TIME]);
}

void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind, 
                     double** val)
{
    fprintf(stdout, "Expanding symmetric matrix ... ");
    int nnz = *nnz_;

    // first, count off-diagonal non-zeros
    int not_diag = 0;
    for(int i = 0; i < nnz; i++) {
        if((*row_ind)[i] != (*col_ind)[i]) {
            not_diag++;
        }
    }

    int* _row_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
    assert(_row_ind);
    int* _col_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
    assert(_col_ind);
    double* _val = (double*) malloc(sizeof(double) * (nnz + not_diag));
    assert(_val);

    memcpy(_row_ind, *row_ind, sizeof(int) * nnz);
    memcpy(_col_ind, *col_ind, sizeof(int) * nnz);
    memcpy(_val, *val, sizeof(double) * nnz);
    int index = nnz;
    for(int i = 0; i < nnz; i++) {
        if((*row_ind)[i] != (*col_ind)[i]) {
            _row_ind[index] = (*col_ind)[i];
            _col_ind[index] = (*row_ind)[i];
            _val[index] = (*val)[i];
            index++;
        }
    }
    assert(index == (nnz + not_diag));

    free(*row_ind);
    free(*col_ind);
    free(*val);

    *row_ind = _row_ind;
    *col_ind = _col_ind;
    *val = _val;
    *nnz_ = nnz + not_diag;

    fprintf(stdout, "done\n");
    fprintf(stdout, "  Total # of non-zeros is %d\n", nnz + not_diag);
}

void init_locks(omp_lock_t** locks, int m)
{
    omp_lock_t* _locks = (omp_lock_t*) malloc(sizeof(omp_lock_t) * m);
    assert(_locks);
    for(int i = 0; i < m; i++) {
        omp_init_lock(&(_locks[i]));
    }
    *locks = _locks;
}

void destroy_locks(omp_lock_t* locks, int m)
{
    assert(locks);
    for(int i = 0; i < m; i++) {
        omp_destroy_lock(&(locks[i]));
    }
    free(locks);
}