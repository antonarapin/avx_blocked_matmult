#include <immintrin.h>



int main(){
	double A[]={1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
	double B[]={1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16};
	double C[8];
	double *a = A;	
	double *b = B;
	double *c = C;
	
	double *bc1=(double*)malloc(4*sizeof(double));
	double *bc2=(double*)malloc(4*sizeof(double));
	double *bc3=(double*)malloc(4*sizeof(double));
	double *bc4=(double*)malloc(4*sizeof(double));
	bc1 = (double[]){*(b),*(b+1),*(b+2),*(b+3)};
	bc2 = (double[]){*(b+4),*(b+5),*(b+6),*(b+7)};
	bc3 = (double[]){*(b+8),*(b+9),*(b+10),*(b+11)};
	bc4 = (double[]){*(b+12),*(b+13),*(b+14),*(b+15)};
	
	double *ar1=(double*)malloc(4*sizeof(double));
        double *ar2=(double*)malloc(4*sizeof(double));
        double *ar3=(double*)malloc(4*sizeof(double));
        double *ar4=(double*)malloc(4*sizeof(double));
	ar1 = (double[]){*(a),*(a+4),*(a+8),*(a+12)};
	ar2 = (double[]){*(a+1),*(a+5),*(a+9),*(a+13)};
	ar3 = (double[]){*(a+2),*(a+6),*(a+10),*(a+14)};
	ar4 = (double[]){*(a+3),*(a+7),*(a+11),*(a+15)};
	

	double *bc=(double*)malloc(4*sizeof(double));
	bc = (double[]){*(b),*(b+4),*(b+8),*(b+12)};
	__m256d ar1 = _mm256_loadu_pd(a);	
	__m256d bc1 = _mm256_loadu_pd(bc);
	__m256d r = _mm256_mul_pd(ar1,bc1);	
	double* res = (double*)&r;
	printf("%f %f %f %f\n",res[0], res[1], res[2], res[3]);
	return 0;
}
