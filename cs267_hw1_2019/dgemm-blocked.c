const char *dgemm_desc = "Simple blocked dgemm.";

#define BLOCK_SIZE 120

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a, b) (((a) < (b))? (a) : (b))

#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>

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

static void do_block_unroll(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; i += 2) {
        //For each column j of B
        for (int j = 0; j < N; j += 2) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            // Compute C(i+1,j)
            double ci1j = C[i + 1 + j * lda];
            // Compute C(i,j+1)
            double cij1 = C[i + (j + 1) * lda];
            // Compute C(i+1,j+1)
            double ci1j1 = C[(i + 1) + (j + 1) * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
                ci1j += A[i + 1 + k * lda] * B[k + j * lda];
                cij1 += A[i + k * lda] * B[k + (j + 1) * lda];
                ci1j1 += A[i + 1 + k * lda] * B[k + (j + 1) * lda];
            }
            C[i + j * lda] = cij;
            C[i + 1 + j * lda] = ci1j;
            C[i + (j + 1) * lda] = cij1;
            C[(i + 1) + (j + 1) * lda] = ci1j1;
        }
    }
}

static void do_block_unroll_transpose(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; i += 2) {
        //For each column j of B
        for (int j = 0; j < N; j += 2) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            // Compute C(i+1,j)
            double ci1j = C[i + 1 + j * lda];
            // Compute C(i,j+1)
            double cij1 = C[i + (j + 1) * lda];
            // Compute C(i+1,j+1)
            double ci1j1 = C[(i + 1) + (j + 1) * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[k + i * lda] * B[k + j * lda];
                ci1j += A[k + (i + 1) * lda] * B[k + j * lda];
                cij1 += A[k + i * lda] * B[k + (j + 1) * lda];
                ci1j1 += A[k + (i + 1) * lda] * B[k + (j + 1) * lda];
            }
            C[i + j * lda] = cij;
            C[i + 1 + j * lda] = ci1j;
            C[i + (j + 1) * lda] = cij1;
            C[(i + 1) + (j + 1) * lda] = ci1j1;
        }
    }
}

static void do_block_unroll_transpose_fix(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    double cij;
    double ci1j;
    double cij1;
    double ci1j1;

    double ci2j;
    double ci2j1;
    double ci2j2;
    double ci1j2;
    double cij2;

//    double ci3j, ci3j1, ci3j2, ci3j3, ci2j3, ci1j3, cij3;

    double t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23;
    int i, j, k;

    for (i = 0; i < M / 3 * 3; i += 3) {
        //For each column j of B
        for (j = 0; j < N / 3 * 3; j += 3) {
            // Compute C(i,j)
            cij = C[i + j * lda];
            // Compute C(i+1,j)
            ci1j = C[i + 1 + j * lda];
            // Compute C(i,j+1)
            cij1 = C[i + (j + 1) * lda];
            // Compute C(i+1,j+1)
            ci1j1 = C[(i + 1) + (j + 1) * lda];


            //Compute C(i+2, j)
            ci2j = C[(i + 2) + (j) * lda];
            //Compute C(i+2, j+1)
            ci2j1 = C[(i + 2) + (j + 1) * lda];
            //Compute C(i+2, j+2)
            ci2j2 = C[(i + 2) + (j + 2) * lda];
            //Compute C(i+1, j+2)
            ci1j2 = C[(i + 1) + (j + 2) * lda];
            //Compute C(i, j+2)
            cij2 = C[(i) + (j + 2) * lda];


//            //Compute C(i+3, j)
//            ci3j = C[(i + 3) + (j) * lda];
//            //Compute C(i+3, j+1)
//            ci3j1 = C[(i + 3) + (j + 1) * lda];
//            //Compute C(i+3, j+2)
//            ci3j2 = C[(i + 3) + (j + 2) * lda];
//            //Compute C(i+3, j+3)
//            ci3j3 = C[(i + 3) + (j + 3) * lda];
//            //Compute C(i+2, j+3)
//            ci2j3 = C[(i + 2) + (j + 3) * lda];
//            //Compute C(i+1, j+3)
//            ci1j3 = C[(i + 1) + (j + 3) * lda];
//            //Compute C(i, j+3)
//            cij3 = C[(i) + (j + 3) * lda];


            for (k = 0; k < K/4 * 4; k += 4) {
                //1st Row
                t0 = A[k + i * lda];
                //1st Col
                t1 = B[k + j * lda];
                //2nd Row
                t2 = A[k + (i + 1) * lda];
                //2nd Col
                t3 = B[k + (j + 1) * lda];
                //3rd Row
                t4 = A[k + (i + 2) * lda];
                //3rd Col
                t5 = B[k + (j + 2) * lda];


                //1st Row
                t6 = A[k + 1 + i * lda];
                //1st Col
                t7 = B[k + 1 + j * lda];
                //2nd Row
                t8 = A[k + 1 + (i + 1) * lda];
                //2nd Col
                t9 = B[k + 1 + (j + 1) * lda];
                //3rd Row
                t10 = A[k + 1 + (i + 2) * lda];
                //3rd Col
                t11 = B[k + 1 + (j + 2) * lda];


                //1st Row
                t12 = A[k + 2 + i * lda];
                //1st Col
                t13 = B[k + 2 + j * lda];
                //2nd Row
                t14 = A[k + 2 + (i + 1) * lda];
                //2nd Col
                t15 = B[k + 2 + (j + 1) * lda];
                //3rd Row
                t16 = A[k + 2 + (i + 2) * lda];
                //3rd Col
                t17 = B[k + 2 + (j + 2) * lda];

                //1st Row
                t18 = A[k + 3 + i * lda];
                //1st Col
                t19 = B[k + 3 + j * lda];
                //2nd Row
                t20 = A[k + 3 + (i + 1) * lda];
                //2nd Col
                t21 = B[k + 3 + (j + 1) * lda];
                //3rd Row
                t22 = A[k + 3 + (i + 2) * lda];
                //3rd Col
                t23 = B[k + 3 + (j + 2) * lda];






//
//                t6 = A[k + (i + 3) * lda];
//                //3rd Col
//                t7 = B[k + (j + 3) * lda];


                cij += t0 * t1 + t6*t7 + t12 * t13 + t18*t19;
                ci1j += t2 * t1 + t8*t7 + t13 * t14 + t19*t20;
                cij1 += t0 * t3 + t6 * t9 + t12*t15 + t18*t21;
                ci1j1 += t2 * t3 + t8*t9 + t14*t15 + t20*t21;

                ci2j += t4 * t1 + t10*t7 + t16*t13 + t22*t19;
                ci2j1 += t4 * t3 + t10*t9 + t16*t15 + t22*t21;
                ci2j2 += t5 * t4 + t11*t10 + t17*t16 + t23*t22;
                ci1j2 += t2 * t5 + t8*t11 + t14*t17 + t20*t23;
                cij2 += t0 * t5 + t6*t11 + t12*t17 + t18*t23;

//                ci3j += t6 * t1;
//                ci3j1 += t6 * t3;
//                ci3j2 += t6 * t5;
//                ci3j3 += t6 * t7;
//                ci2j3 += t4 * t7;
//                ci1j3 += t2 * t7;
//                cij3 += t0 * t7;


            }


            for(int k = K/4*4; k < K; k++){
                //1st Row
                t0 = A[k + i * lda];
                //1st Col
                t1 = B[k + j * lda];
                //2nd Row
                t2 = A[k + (i + 1) * lda];
                //2nd Col
                t3 = B[k + (j + 1) * lda];
                //3rd Row
                t4 = A[k + (i + 2) * lda];
                //3rd Col
                t5 = B[k + (j + 2) * lda];

                cij += t0 * t1;
                ci1j += t2 * t1;
                cij1 += t0 * t3;
                ci1j1 += t2 * t3;

                ci2j += t4 * t1;
                ci2j1 += t4 * t3;
                ci2j2 += t5 * t4;
                ci1j2 += t2 * t5;
                cij2 += t0 * t5;
            }


            C[i + j * lda] = cij;
            C[i + 1 + j * lda] = ci1j;
            C[i + (j + 1) * lda] = cij1;
            C[(i + 1) + (j + 1) * lda] = ci1j1;

            C[(i + 2) + (j) * lda] = ci2j;
            C[(i + 2) + (j + 1) * lda] = ci2j1;
            C[(i + 2) + (j + 2) * lda] = ci2j2;
            C[(i + 1) + (j + 2) * lda] = ci1j2;
            C[(i) + (j + 2) * lda] = cij2;


//            C[(i + 3) + (j) * lda] = ci3j;
//            C[(i + 3) + (j + 1) * lda] = ci3j1;
//            C[(i + 3) + (j + 2) * lda] = ci3j2;
//            C[(i + 3) + (j + 3) * lda] = ci3j3;
//            C[(i + 2) + (j + 3) * lda] = ci2j3;
//            C[(i + 1) + (j + 3) * lda] = ci1j3;
//            C[(i) + (j + 3) * lda] = cij3;


        }
//        The odd row of matrix B, this should only have ONE iteration!
        for (j = N / 3 * 3; j < N; ++j) {
            cij = C[i + j * lda];
            ci1j = C[i + 1 + j * lda];
            ci2j = C[i + 2 + j * lda];
//            ci3j = C[i + 3 + j * lda];
            for (k = 0; k < K; ++k) {
                cij += A[k + i * lda] * B[k + j * lda];
                ci1j += A[k + (i + 1) * lda] * B[k + j * lda];
                ci2j += A[k + (i + 2) * lda] * B[k + j * lda];
//                ci3j += A[k + (i + 3) * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
            C[i + 1 + j * lda] = ci1j;
            C[i + 2 + j * lda] = ci2j;
//            C[i + 3 + j * lda] = ci3j;

        }

    }


//    printMatrix(C, lda, "Matrix C post");
    //The odd row of matrix A, this should only have ONE iteration!
    for (i = M / 3 * 3; i < M; ++i) {
        //For each column j of B
        for (j = 0; j < N; ++j) {
            cij = C[i + j * lda];
            for (k = 0; k < K; ++k) {
                cij += A[k + i * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }

}


static void do_block_unroll_transpose_vect1(int lda, int M, int N, int K, double *A, double *B, double *C) {

    __m256d vectorA1;
    __m256d vectorB1;
    __m256d vectorA2;
    __m256d vectorB2;
    __m256d vectorCij;
    __m256d vectorCi1j;
    __m256d vectorCij1;
    __m256d vectorCi1j1;
    // For each row i of A
    for (int i = 0; i < M; i += 2) {
        //For each column j of B
        for (int j = 0; j < N; j += 2) {
            // Compute C(i,j)
            double cij[4] = {C[i + j * lda], 0, 0, 0};
            vectorCij = _mm256_loadu_pd(cij);
            // Compute C(i+1,j)
            double ci1j[4] = {C[i + 1 + j * lda], 0, 0, 0};
            vectorCi1j = _mm256_loadu_pd(ci1j);
            // Compute C(i,j+1)
            double cij1[4] = {C[i + (j + 1) * lda], 0, 0, 0};
            vectorCij1 = _mm256_loadu_pd(cij1);
            // Compute C(i+1,j+1)
            double ci1j1[4] = {C[(i + 1) + (j + 1) * lda], 0, 0, 0};
            vectorCi1j1 = _mm256_loadu_pd(ci1j1);
            for (int k = 0; k < K / 4 * 4; k += 4) {
                vectorA1 = _mm256_loadu_pd(A + (k + i * lda));
                vectorB1 = _mm256_loadu_pd(B + (k + j * lda));
                vectorA2 = _mm256_loadu_pd(A + (k + (i + 1) * lda));
                vectorB2 = _mm256_loadu_pd(B + (k + (j + 1) * lda));
                vectorCij = _mm256_add_pd(vectorCij, _mm256_mul_pd(vectorA1, vectorB1));
                vectorCi1j = _mm256_add_pd(vectorCi1j, _mm256_mul_pd(vectorA2, vectorB1));
                vectorCij1 = _mm256_add_pd(vectorCij1, _mm256_mul_pd(vectorA1, vectorB2));
                vectorCi1j1 = _mm256_add_pd(vectorCi1j1, _mm256_mul_pd(vectorA2, vectorB2));
                //cij += A[k + i * lda] * B[k + j * lda];
                //ci1j += A[k + (i + 1)* lda] * B[k + j * lda];
                //cij1 += A[k + i * lda] * B[k + (j + 1) * lda];
                //ci1j1 += A[k + (i + 1) * lda] * B[k + (j + 1) * lda];
            }

            _mm256_store_pd(cij, vectorCij);
            _mm256_store_pd(ci1j, vectorCi1j);
            _mm256_store_pd(cij1, vectorCij1);
            _mm256_store_pd(ci1j1, vectorCi1j1);

            for (int k = K / 4 * 4; k < K; k++) {
                cij[0] += A[k + i * lda] * B[k + j * lda];
                ci1j[0] += A[k + (i + 1) * lda] * B[k + j * lda];
                cij1[0] += A[k + i * lda] * B[k + (j + 1) * lda];
                ci1j1[0] += A[k + (i + 1) * lda] * B[k + (j + 1) * lda];
            }

            C[i + j * lda] = cij[0] + cij[1] + cij[2] + cij[3];
            C[i + 1 + j * lda] = ci1j[0] + ci1j[1] + ci1j[2] + ci1j[3];
            C[i + (j + 1) * lda] = cij1[0] + cij1[1] + cij1[2] + cij1[3];
            C[(i + 1) + (j + 1) * lda] = ci1j1[0] + ci1j1[1] + ci1j1[2] + ci1j1[3];
        }
    }
}


static void do_block_transpose_vect(int lda, int M, int N, int K, double *A, double *B, double *C) {

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
                vectorA = _mm256_loadu_pd(A + (k + i * lda));
                vectorB = _mm256_loadu_pd(B + (k + j * lda));

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

static void do_block_unroll_transpose_vect2(int lda, int M, int N, int K, double *A, double *B, double *C) {

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
                vectorA = _mm256_loadu_pd(A + (k + i * lda));
                vectorB = _mm256_loadu_pd(B + (k + j * lda));

                //A * B
                vectorA = _mm256_mul_pd(vectorA, vectorB);
                vectorC = _mm256_add_pd(vectorC, vectorA);

                //Loads buffer to vector, loads B to vector
                vectorA = _mm256_loadu_pd(A + (k + i * lda + 4));
                vectorB = _mm256_loadu_pd(B + (k + j * lda + 4));

                //A * B
                vectorA = _mm256_mul_pd(vectorA, vectorB);
                vectorC = _mm256_add_pd(vectorC, vectorA);

                //Loads buffer to vector, loads B to vector
                vectorA = _mm256_loadu_pd(A + (k + i * lda + 8));
                vectorB = _mm256_loadu_pd(B + (k + j * lda + 8));

                //A * B
                vectorA = _mm256_mul_pd(vectorA, vectorB);
                vectorC = _mm256_add_pd(vectorC, vectorA);

                //Loads buffer to vector, loads B to vector
                vectorA = _mm256_loadu_pd(A + (k + i * lda + 12));
                vectorB = _mm256_loadu_pd(B + (k + j * lda + 12));

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
    double *B_aligned = (double *) malloc(sizeof(double) * lda * lda);
//    double *C_aligned = (double *) malloc(sizeof(double) * lda * lda);

    //Transpose A -> row-major format
    for (int i = 0; i < lda; i++) {
        for (int j = 0; j < lda; j++) {
            A_t[j + lda * i] = A[i + lda * j];
            B_aligned[i + lda * j] = B[i + lda * j];
//            C_aligned[i + lda * j] = C[i + lda * j];
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
                do_block_unroll_transpose_fix(lda, M, N, K, A_t + k + i * lda, B_aligned + k + j * lda,
                                              C + i + j * lda);

//                do_block_unroll_transpose(lda, M, N, K, A_t + k + i * lda, B + k + j * lda, C + i + j * lda);
                //do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
        }
    }

//    for (int i = 0; i < lda; i++) {
//        for (int j = 0; j < lda; j++) {
//
//            C[i + lda * j] = C_aligned[i + lda * j];
//
//        }
//    }

    free(A_t);
    free(B_aligned);
//    free(C_aligned);
}
