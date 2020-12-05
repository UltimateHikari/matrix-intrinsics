#include <cblas.h>
#include <time.h>
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

int main(){
	int N=1,M;
	struct timespec start,lap1,lap2,lap3,end;
	ifstream fin("input_l.txt");
	ofstream fout("output.txt");fout.close();
						// srand(time(NULL));
						// Matrix A(2048);
						// random_matrix(&A);
						// A.out();
	fin >> N >> M;
	Matrix A(N), R(N), B(N,true), BA(N), RR(N);
	for(int i = 0; i < N*N; ++i){
		fin >> A.m[i];
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	trivial_mult_opt(&BA, &A, &B);
	clock_gettime(CLOCK_MONOTONIC_RAW, &lap1);
	vector_mult(&A,&B,&BA);
	clock_gettime(CLOCK_MONOTONIC_RAW, &lap2);
	vector_mult_opt(&A,&B,&BA);
	clock_gettime(CLOCK_MONOTONIC_RAW, &lap3);
	cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,
		N,N,N,1.0,A.m,N,B.m,N, 0.0,BA.m,N);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);


	time_delta("optimized  : ",&start, &lap1);
	time_delta("vect_core  : ",&lap1, &lap2);
	time_delta("corereorder: ",&lap2, &lap3);
	time_delta("cblas      : ",&lap3, &end);
	//A.out();
	//B.out();
	BA.out();
	fin.close();
	return 0;
}