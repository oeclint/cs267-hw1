const char *dgemm_desc = "Simple blocked dgemm.";

#define BLOCK_SIZE 500

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a, b) (((a) < (b))? (a) : (b))

#include <immintrin.h>
#include <stdlib.h>

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}


static void do_block_vectorized(int lda, int M, int N, int K, double *A, double *B, double *C) {

    __m256d vectorA;
    __m256d vectorB;
    __m256d vectorC;

    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {

            double cij[4] = {C[i + j * lda], 0, 0, 0};
            vectorC = _mm256_loadu_pd(cij);
            for (int k = 0; k < K / 4 * 4; k += 4) {


                //Loads buffer to vector, loads B to vector
                vectorA = _mm256_loadu_pd(A + (k + i*lda));
                vectorB = _mm256_loadu_pd(B + (k+j*lda));

                //A * B
                vectorA = _mm256_mul_pd(vectorA, vectorB);
                vectorC = _mm256_add_pd(vectorC, vectorA);

            }

            _mm256_store_pd(cij, vectorC);

            for (int k = K / 4 * 4; k < K; k++) {
                cij[0] += A[k + i * lda] * B[k + j * lda];
            }

            C[i + j * lda] = cij[0] + cij[1] + cij[2] + cij[3];
        }
    }
}

static void do_block_vectorized_unroll4(int lda, int M, int N, int K, double *A, double *B, double *C) {

    __m256d vectorA;
    __m256d vectorB;
    __m256d vectorC;

    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {

            double cij[4] = {C[i + j * lda], 0, 0, 0};
            vectorC = _mm256_loadu_pd(cij);
            for (int k = 0; k < K / 16 * 16; k += 16) {


                //Loads buffer to vector, loads B to vector
                vectorA = _mm256_loadu_pd(A + (k + i*lda));
                vectorB = _mm256_loadu_pd(B + (k+j*lda));

                //A * B
                vectorA = _mm256_mul_pd(vectorA, vectorB);
                vectorC = _mm256_add_pd(vectorC, vectorA);

                //Loads buffer to vector, loads B to vector
                vectorA = _mm256_loadu_pd(A + (k + i*lda+4));
                vectorB = _mm256_loadu_pd(B + (k+j*lda+4));

                //A * B
                vectorA = _mm256_mul_pd(vectorA, vectorB);
                vectorC = _mm256_add_pd(vectorC, vectorA);

                //Loads buffer to vector, loads B to vector
                vectorA = _mm256_loadu_pd(A + (k + i*lda+8));
                vectorB = _mm256_loadu_pd(B + (k+j*lda+8));

                //A * B
                vectorA = _mm256_mul_pd(vectorA, vectorB);
                vectorC = _mm256_add_pd(vectorC, vectorA);

                //Loads buffer to vector, loads B to vector
                vectorA = _mm256_loadu_pd(A + (k + i*lda+12));
                vectorB = _mm256_loadu_pd(B + (k+j*lda+12));

                //A * B
                vectorA = _mm256_mul_pd(vectorA, vectorB);
                vectorC = _mm256_add_pd(vectorC, vectorA);

            }

            _mm256_store_pd(cij, vectorC);

            for (int k = K / 16 * 16; k < K; k++) {
                cij[0] += A[k + i * lda] * B[k + j * lda];
            }

            C[i + j * lda] = cij[0] + cij[1] + cij[2] + cij[3];
        }
    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {

    double *A_t = (double *) malloc(sizeof(double) * lda * lda);

    //Transpose A -> row-major format
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < lda; j++) {
            A_t[j + lda * i] = A[i + lda * j];
        }
    }


    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min (BLOCK_SIZE, lda - i);
                int N = min (BLOCK_SIZE, lda - j);
                int K = min (BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block_vectorized_unroll4(lda, M, N, K, A_t + k + i * lda, B + k + j * lda, C + i + j * lda);

                //do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
        }
    }
    free(A_t);
}
