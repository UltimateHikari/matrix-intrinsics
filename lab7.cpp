#include <cblas.h>
#include <time.h>
#include <cmath>
#include <cstring>
#include "mult.h"
using namespace std;

void time_delta(
	const char* head,
	struct timespec * start,
	struct timespec * end)
{
	std::cout << head << end->tv_sec - start->tv_sec 
	+ 1e-9*(end->tv_nsec - start->tv_nsec) << endl;
}

void random_matrix(Matrix *A){
	srand(time(NULL));
	for(int i = 0; i < A->n*A->n; ++i){
		A->m[i] = rand();
	}
}

void trivial_b(Matrix* B, Matrix* A){
	int size = A->n;
	float* A1s = (float*)calloc(size, sizeof(float));
	float* Ainfs = (float*)calloc(size, sizeof(float));
	for(int i = 0; i< size; ++i){
		float* a = A->m + i*size;
		for(int j = 0; j< size; ++j){
			float cell = fabs(a[j]);
			A1s[j] += cell;
			Ainfs[i] += cell;
		}
	}
	float A1 = 0, Ainf = 0;
	for(int i = 0; i < size; ++i){
		if(A1s[i] > A1)
			A1 = A1s[i];
	}
	for(int i = 0; i < size; ++i){
		if(Ainfs[i] > Ainf)
			Ainf = Ainfs[i];
	}
	float Adiv = A1*Ainf;
	//unoptimal in any case
	for(int i = 0; i < size; ++i){
		float* b = B->m + i*size;
		for(int j = 0; j< size; ++j){
			b[j] = A->m[j*size + i]/Adiv;
		}
	}
	free(A1s); free(Ainfs);
}

void ptr_swap(Matrix** a, Matrix** b){
	Matrix* c = *a;
	*a = *b;
	*b = c;
}

void trivial_invert(int N, int M,Matrix* A, Matrix* A1){
	Matrix R(N,true), B(N,true),
	C(N,true), R1(N), R2(N);
	trivial_b(&B, A);
	trivial_mult_opt(A1, &B, A); //reusing as BA
	trivial_sub(&R, A1);
	Matrix* rcur = &R1;
	Matrix* rnext = &R2;
	trivial_sum(rcur, &R);
	trivial_sum(&C, rcur);
	for(int i = 2; i < M; ++i){
		trivial_mult_opt(rnext, rcur, &R);
		trivial_sum(&C, rnext);
		ptr_swap(&rnext, &rcur);
		memset(rnext->m, 0, sizeof(float)*N*N);
	}
	memset(A1->m, 0, sizeof(float)*N*N);
	trivial_mult_opt(A1, &C, &B);
}

void vect_invert(int N, int M,Matrix* A, Matrix* A1){
	Matrix R(N,true), B(N,true),
	C(N,true), R1(N), R2(N);
	trivial_b(&B, A);
	vector_mult_opt(&B, A, A1); //reusing as BA
	trivial_sub(&R, A1);
	Matrix* rcur = &R1;
	Matrix* rnext = &R2;
	trivial_sum(rcur, &R);
	trivial_sum(&C, rcur);
	for(int i = 2; i < M; ++i){
		vector_mult_opt(rcur, &R,rnext);
		trivial_sum(&C, rnext);
		ptr_swap(&rnext, &rcur);
		memset(rnext->m, 0, sizeof(float)*N*N);
	}
	memset(A1->m, 0, sizeof(float)*N*N);
	vector_mult_opt(&C, &B,A1);
}

void blas_invert(int N, int M,Matrix* A, Matrix* A1){
	Matrix R(N,true), B(N,true),
	C(N,true), R1(N), R2(N);
	trivial_b(&B, A);
	cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,
			N,N,N,1.0,B.m,N,A->m,N, 0.0,A1->m,N);
	//vector_mult_opt(&B, A, A1); //reusing as BA
	trivial_sub(&R, A1);
	Matrix* rcur = &R1;
	Matrix* rnext = &R2;
	trivial_sum(rcur, &R);
	trivial_sum(&C, rcur);
	for(int i = 2; i < M; ++i){
		cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,
			N,N,N,1.0,rcur->m,N,R.m,N, 0.0,rnext->m,N);
		//vector_mult_opt(rcur, &R,rnext);
		trivial_sum(&C, rnext);
		ptr_swap(&rnext, &rcur);
		memset(rnext->m, 0, sizeof(float)*N*N);
	}
	memset(A1->m, 0, sizeof(float)*N*N);
	vector_mult_opt(&C, &B,A1);
}

int main(){
	int N=1,M;
	struct timespec start,lap1,lap2,lap3,end;
	ifstream fin("input.txt");
	ofstream fout("output.txt");fout.close();
						// srand(time(NULL));
						// Matrix A(2048);
						// random_matrix(&A);
						// A.out();
	fin >> N >> M;
	// (N,true) == I
	Matrix A(N), Ct(N),Cv(N),Cb(N);
	for(int i = 0; i < N*N; ++i){
		fin >> A.m[i];
	}
	//inreference between? wtf?
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	trivial_invert(N,M,&A, &Ct);
	clock_gettime(CLOCK_MONOTONIC_RAW, &lap1);
	vect_invert(N,M,&A,&Cv);
	clock_gettime(CLOCK_MONOTONIC_RAW, &lap2);
	blas_invert(N,M,&A,&Cb);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	time_delta("optimized  : ",&start, &lap1);
	time_delta("corereorder: ",&lap1, &lap2);
	time_delta("cblas      : ",&lap2, &end);

	// clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	// trivial_mult_opt(&BA, &A, &B);
	// clock_gettime(CLOCK_MONOTONIC_RAW, &lap1);
	// vector_mult(&A,&B,&BA);
	// clock_gettime(CLOCK_MONOTONIC_RAW, &lap2);
	// vector_mult_opt(&A,&B,&BA);
	// clock_gettime(CLOCK_MONOTONIC_RAW, &lap3);
	// cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,
	// 	N,N,N,1.0,A.m,N,B.m,N, 0.0,BA.m,N);
	// clock_gettime(CLOCK_MONOTONIC_RAW, &end);


	//time_delta("optimized  : ",&start, &lap1);
	// time_delta("vect_core  : ",&lap1, &lap2);
	// time_delta("corereorder: ",&lap2, &lap3);
	// time_delta("cblas      : ",&lap3, &end);

	Ct.out();
	Cv.out();
	Cb.out();
	fin.close();
	return 0;
}