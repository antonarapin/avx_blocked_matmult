/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <immintrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 4
#define K_SIZE 900
#define M_SIZE 4
#define N_SIZE 8
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  if((M==M_SIZE && N==N_SIZE)){

    __m256d c1,c2,c3,c4,c5,c6,c7,c8,a,b1,b2,b3,b4,b5,b6,b7,b8;
    c1 = _mm256_loadu_pd(C+0*lda);
    c2 = _mm256_loadu_pd(C+1*lda);
    c3 = _mm256_loadu_pd(C+2*lda);
    c4 = _mm256_loadu_pd(C+3*lda);
    c5 = _mm256_loadu_pd(C+4*lda);
    c6 = _mm256_loadu_pd(C+5*lda);
    c7 = _mm256_loadu_pd(C+6*lda);
    c8 = _mm256_loadu_pd(C+7*lda);
    //c9 = _mm256_loadu_pd(C+8*lda);
    //c10 = _mm256_loadu_pd(C+9*lda);
    //c11 = _mm256_loadu_pd(C+10*lda);
    //c12 = _mm256_loadu_pd(C+11*lda);
    for(int i = 0; i < K; i++){
      a = _mm256_loadu_pd(A+i*lda);
      b1=_mm256_set1_pd(B[i+0*lda]);
      b2=_mm256_set1_pd(B[i+1*lda]);
      b3=_mm256_set1_pd(B[i+2*lda]);
      b4=_mm256_set1_pd(B[i+3*lda]);
      b5=_mm256_set1_pd(B[i+4*lda]);
      b6=_mm256_set1_pd(B[i+5*lda]);
      b7=_mm256_set1_pd(B[i+6*lda]);
      b8=_mm256_set1_pd(B[i+7*lda]);
      //b9=_mm256_set1_pd(B[i+8*lda]);
      //b10=_mm256_set1_pd(B[i+9*lda]);
      //b11=_mm256_set1_pd(B[i+10*lda]);
      //b12=_mm256_set1_pd(B[i+11*lda]);
      c1=_mm256_add_pd(c1,_mm256_mul_pd(a,b1));
      c2=_mm256_add_pd(c2,_mm256_mul_pd(a,b2));
      c3=_mm256_add_pd(c3,_mm256_mul_pd(a,b3));
      c4=_mm256_add_pd(c4,_mm256_mul_pd(a,b4));
      c5=_mm256_add_pd(c5,_mm256_mul_pd(a,b5));
      c6=_mm256_add_pd(c6,_mm256_mul_pd(a,b6));
      c7=_mm256_add_pd(c7,_mm256_mul_pd(a,b7));
      c8=_mm256_add_pd(c8,_mm256_mul_pd(a,b8));
      //c9=_mm256_add_pd(c9,_mm256_mul_pd(a,b9));
      //c10=_mm256_add_pd(c10,_mm256_mul_pd(a,b10));
      //c11=_mm256_add_pd(c11,_mm256_mul_pd(a,b11));
      //c12=_mm256_add_pd(c12,_mm256_mul_pd(a,b12));
    }
    _mm256_storeu_pd(C+0*lda,c1);
    _mm256_storeu_pd(C+1*lda,c2);
    _mm256_storeu_pd(C+2*lda,c3);
    _mm256_storeu_pd(C+3*lda,c4);
    _mm256_storeu_pd(C+4*lda,c5);
    _mm256_storeu_pd(C+5*lda,c6);
    _mm256_storeu_pd(C+6*lda,c7);
    _mm256_storeu_pd(C+7*lda,c8);
    //_mm256_storeu_pd(C+8*lda,c9);
    //_mm256_storeu_pd(C+9*lda,c10);
    //_mm256_storeu_pd(C+10*lda,c11);
    //_mm256_storeu_pd(C+11*lda,c12);
  }else if(M==M_SIZE){
    __m256d a;
    __m256d cs[N];
    for(int t=0;t<N;t++){
      cs[t] = _mm256_loadu_pd(C+t*lda);
    }
    for(int i = 0; i < K; i++){
      a = _mm256_loadu_pd(A+i*lda);
      __m256d bs[N];
      for(int t=0;t<N;t++){
      	bs[t]=_mm256_set1_pd(B[i+t*lda]);
      }
      for(int t=0;t<N;t++){
	cs[t]=_mm256_add_pd(cs[t],_mm256_mul_pd(a,bs[t]));
      }

    }
    for(int t=0;t<N;t++){
      _mm256_storeu_pd(C+t*lda,cs[t]);
    }
  }else{
    
    for (int i = 0; i < M; ++i){
      for (int j = 0; j < N; ++j) 
      {
        double cij = C[i+j*lda];
        for (int k = 0; k < K; ++k){
	  cij += A[i+k*lda] * B[k+j*lda];
        }
        C[i+j*lda] = cij;
      }
    }
    
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += M_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += N_SIZE)
      /* Accumulate block dgemms into block of C */
      for(int k = 0 ; k < lda; k+=K_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int N = min (N_SIZE, lda-j);
	int M = min (M_SIZE, lda-i);
	int K = min (K_SIZE, lda-k);
	/* Perform individual block dgemm */
	
	do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
}
