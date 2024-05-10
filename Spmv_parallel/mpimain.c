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
#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

int main(int argc, char** argv) {
    // program info
    usage(argc, argv);
    int num_procs = 0, rank = 0;
    MPI_Init(&argc, &argv);
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    if (rank==0) {
        printf("number of process: %d\n", num_procs);
    }

    uint64_t start_t;
    uint64_t end_t;
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
    ret = mm_read_mtx_crd(matrixName, &m, &n, &nnz, &row_ind, &col_ind, &val, 
                          &matcode);
    check_mm_ret(ret);
    if(is_symmetric) {
        expand_symmetry(m, n, &nnz, &row_ind, &col_ind, &val);
    }


    
    // Convert co-ordinate format to CSR format
    fprintf(stdout, "Converting COO to CSR...");
    unsigned int* csr_row_ptr = NULL; 
    unsigned int* csr_col_ind = NULL;  
    double* csr_vals = NULL; 
    // IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED
    convert_coo_to_csr(row_ind, col_ind, val, m, n, nnz,
                       &csr_row_ptr, &csr_col_ind, &csr_vals);
    fprintf(stdout, "done\n");


    // Load the input vector file
    char vectorName[MAX_FILENAME];
    strcpy(vectorName, argv[2]);
    fprintf(stdout, "Vector file name: %s ... ", vectorName);
    double* vector_x;
    unsigned int vector_size;
    read_vector(vectorName, &vector_x, &vector_size);
    assert(n == vector_size);
    fprintf(stdout, "file loaded\n");

    omp_lock_t* writelock; 
    init_locks(&writelock, m);

    //MPI_COO
    double *res_coo_MPI = (double*) malloc(sizeof(double) * m);;
    assert(res_coo_MPI);
    double *res_coo_local = (double*) malloc(sizeof(double) * m);
    int sum = 0;
    int remain = nnz%num_procs;
    int workload = nnz/num_procs;
    workload++;
    int *scount = (int*) malloc(sizeof(int) * num_procs);
    int *displa = (int*) malloc(sizeof(int) * num_procs);
    unsigned int *local_col = (unsigned int*) malloc(sizeof(unsigned int) * workload);
    unsigned int *local_row = (unsigned int*) malloc(sizeof(unsigned int) * workload);
    double *local_val = (double*) malloc(sizeof(double) * workload);
    
    if (rank==0) {
        fprintf(stdout, "Calculating COO SpMV MPI... ");
    }
    start_t = ReadTSC();
    for (int i=0; i<num_procs; i++) {
        scount[i] = nnz/num_procs;
        if (remain > 0) {
            scount[i]++;
            remain--;
        }
        displa[i] = sum;
        sum += scount[i];
    }
    
    MPI_SAFE_CALL(MPI_Scatterv(col_ind, scount, displa, MPI_INT, local_col, workload, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Scatterv(row_ind, scount, displa, MPI_INT, local_row, workload, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Scatterv(val, scount, displa, MPI_DOUBLE, local_val, workload, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        // IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED 
        spmv_coo(local_row, local_col, local_val, m, n, workload, vector_x, res_coo_local, 
                 writelock);
    }
    MPI_SAFE_CALL(MPI_Reduce(res_coo_local, res_coo_MPI, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

    end_t = ReadTSC();
    if (rank == 0) {
        printf("Time: %f\n", ElapsedTime(end_t - start_t));
        fprintf(stdout, "done\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ///csr_mpi
    double *res_csr_mpi = (double*) malloc(sizeof(double) * m+1);;
    assert(res_csr_mpi);
    int ssum = 0, accu = 0, nnzaccu = 0, gsum = 0;
    int sremain = (m)%num_procs;
    int sworkload = (m)/num_procs;
    sworkload += 3;
    //sentinel
    int *csrscount = (int*) malloc(sizeof(int) * num_procs);
    int *csrdispla = (int*) malloc(sizeof(int) * num_procs);
    int *gcsrscount = (int*) malloc(sizeof(int) * num_procs);
    int *gcsrdispla = (int*) malloc(sizeof(int) * num_procs);
    
    double *local_res_csr = (double*) malloc(sizeof(double) * sworkload-2);
    unsigned int *cslocal_row = (unsigned int*) malloc(sizeof(unsigned int) * sworkload);
    
    if (rank == 0) fprintf(stdout, "Calculating CSR SpMV MPI... ");
    start_t = ReadTSC();
    for (int i=0; i<num_procs; i++) {
        csrscount[i] = (m)/num_procs;
        gcsrscount[i] = (m)/num_procs;
        if (sremain > 0) {
            csrscount[i]++;
            gcsrscount[i]++;
            sremain--;
        }
        
        csrdispla[i] = ssum;
        gcsrdispla[i] = gsum;
        gsum += gcsrscount[i];
        ssum += csrscount[i];
        csrscount[i]++;
    }
    
    
    
    MPI_SAFE_CALL(MPI_Bcast(csr_col_ind, nnz, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Scatterv(csr_row_ptr, csrscount, csrdispla, MPI_INT, cslocal_row, sworkload, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Bcast(csr_vals, nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    
    
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        // IMPLEMENT THIS FUNCTION - MAKE SURE IT'S PARALLELIZED 
        spmv(cslocal_row, csr_col_ind, csr_vals, sworkload, n, nnz, vector_x, local_res_csr);
    }
     
    //MPI_SAFE_CALL(MPI_Gatherv(local_res_csr, sworkload-2, MPI_DOUBLE, res_csr_mpi, gcsrscount, gcsrdispla, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    MPI_SAFE_CALL(MPI_Gather(local_res_csr, sworkload-2, MPI_DOUBLE, res_csr_mpi, sworkload-2, MPI_DOUBLE, 0, MPI_COMM_WORLD));
    end_t = ReadTSC();
    if (rank == 0) {
        printf("Time: %f\n", ElapsedTime(end_t - start_t));
        fprintf(stdout, "done\n");
    }
    



    ///store results
    char resName[MAX_FILENAME];
    strcpy(resName, argv[3]); 
    fprintf(stdout, "Result file name: %s ... ", resName);
    store_result(resName, res_csr_mpi, m);
    // store_result(resName, res_coo, m);
    fprintf(stdout, "file saved\n");
    
    free(csr_row_ptr);
    free(csr_col_ind);
    free(csr_vals);
    free(vector_x);
    //free(res_coo);
    free(res_coo_MPI);
    free(res_coo_local);
    free(scount);
    free(displa);
    free(local_col);
    free(local_row);
    free(local_val);
    free(res_csr_mpi);
    free(gcsrscount);
    free(gcsrdispla);
    free(csrscount);
    free(csrdispla);
    free(local_res_csr);
    free(cslocal_row);
    free(row_ind);
    free(col_ind);
    free(val);
    
    destroy_locks(writelock, m);
    MPI_Finalize();
    
    return 0;
}

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

void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res)
{
      
    
    
    double *resvals = (double*) malloc(sizeof(double) * m);
    //printf("m: %d\n", m);
    #pragma omp parallel for schedule(static)
    for(int i=0; i<m; i++){
        double csrval = 0.0, vecval = 0.0;
        int csr_c = 0, csr_r = 0, start, end;
        double sum = 0.0;
        start = csr_row_ptr[i];
        end = csr_row_ptr[i+1];
        //printf("start: %d, col: %d\n", start, csr_col_ind[start]);
        for(int index=start; index<end; index++){ 
            csr_c = csr_col_ind[index];
            //printf("csr_c: %d\n", csr_c);
            csrval = csr_vals[index];
            vecval = vector_x[csr_c-1];
            sum += csrval * vecval;
        }
        
        resvals[i] = sum;
    
    }
    
    memcpy(res, resvals, sizeof(double) * m);
}

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