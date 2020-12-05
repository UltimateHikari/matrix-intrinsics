#include "mult.h"

using namespace std;

void trivial_sum(Matrix* A, Matrix* B){
	if(A->n != B->n){
		std::cerr << "size mistmatch, aborting\n";
		return;
	}

	int I_MAX =  A->n*A->n;
	for(int i = 0; i < I_MAX; ++i){
		A->m[i] += B->m[i];
	}
}

void trivial_sub(Matrix* A, Matrix* B){
	if(A->n != B->n){
		std::cerr << "size mistmatch, aborting\n";
		return;
	}

	int I_MAX =  A->n*A->n;
	for(int i = 0; i < I_MAX; ++i){
		A->m[i] -= B->m[i];
	}
}

void trivial_mult(Matrix* A, Matrix* B, Matrix* C){
	int I_MAX = A->n;
	for(int i = 0; i < I_MAX; ++i){
		for(int k = 0; k < I_MAX; ++k){
			for(int j = 0; j < I_MAX; ++j){
				A->m[i*I_MAX + k] += B->m[i*I_MAX + j]*C->m[j*I_MAX + k];
			}
		}
	}
}

void trivial_mult_opt(Matrix* A, Matrix* B, Matrix* C){
	int size = A->n;
	//Aik = Bij Cjk
	//sequently caching only b and c row
	for(int i = 0; i < size; ++i){
		float* a = A->m + i*size;
		for(int j = 0; j < size; ++j){
			float b = B->m[i*size + j]; //value not pointer
			float* c = C->m + j*size;
			for(int k = 0; k < size; ++k){
				a[k] += b*c[k];
			}
		}
	}
}

	/*
	* lscpu:
	* i5-4300M
	* L1d cache: 64 KiB
	* L1i cache: 64 KiB
	* L2 cache:  512 KiB
	* L3 cache:  3 MiB
	* avx avx2
	*/

void microcore_4x16(
	int size, float * A, float * B, int ldb, float * C)
{ 
	//not 6x16 `cause 2048 size
	//A*B = C
	__m256 c00 = _mm256_setzero_ps();
	__m256 c10 = _mm256_setzero_ps();
	__m256 c20 = _mm256_setzero_ps();
	__m256 c30 = _mm256_setzero_ps();
	__m256 c01 = _mm256_setzero_ps();
	__m256 c11 = _mm256_setzero_ps();
	__m256 c21 = _mm256_setzero_ps();
	__m256 c31 = _mm256_setzero_ps();
	__m256 b0, b1, a0, a1;
	for(int i = 0; i < size; ++i){
		b0 = _mm256_loadu_ps(B);
		b1 = _mm256_loadu_ps(B + 8);
		a0 = _mm256_set1_ps(A[0*size]);
		a1 = _mm256_set1_ps(A[1*size]);
		c00 = _mm256_fmadd_ps(a0,b0,c00);
		c01 = _mm256_fmadd_ps(a0,b1,c01);
		c10 = _mm256_fmadd_ps(a1,b0,c10);
		c11 = _mm256_fmadd_ps(a1,b1,c11);
		a0 = _mm256_set1_ps(A[2*size]);
		a1 = _mm256_set1_ps(A[3*size]);
		c20 = _mm256_fmadd_ps(a0, b0, c20);
		c21 = _mm256_fmadd_ps(a0, b1, c21);
		c30 = _mm256_fmadd_ps(a1, b0, c30);
		c31 = _mm256_fmadd_ps(a1, b1, c31);
		B += ldb;
		A += 1;
	}
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
	C += size;
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
	C += size;
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + 8, _mm256_add_ps(c21, _mm256_loadu_ps(C + 8)));
	C += size;
	_mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
	_mm256_storeu_ps(C + 8, _mm256_add_ps(c31, _mm256_loadu_ps(C + 8)));
}

void init_res_core(int N, float* C, int size){
	for(int i = 0; i < N; ++i, C+=size){
		_mm256_storeu_ps(C + 0, _mm256_setzero_ps());
		_mm256_storeu_ps(C + 8, _mm256_setzero_ps());
	}
}

void vector_mult(Matrix* A, Matrix* B, Matrix* C){
	int size = A->n;
	if(size % 16 != 0){
		std::cerr << "wrong matrix size, aborting\n";
		return;
	}
	for(int i = 0; i < size; i+=4){
		for(int j = 0; j < size; j+=16){
			init_res_core(4,C->m + i*size + j, size);
			microcore_4x16(size, 
				A->m + i*size, B->m + j, size,
				C->m + i*size + j);
		}
	}
}

void b_reorder_to_16(buf16* buf, float* B, int size){
	float* bufB = buf->d;
	for(int i = 0; i<size; ++i, bufB +=16, B+=size){
		_mm256_storeu_ps(bufB    , _mm256_loadu_ps(B    ));
		_mm256_storeu_ps(bufB + 8, _mm256_loadu_ps(B + 8));
	}
}

void vector_mult_opt(Matrix* A, Matrix* B, Matrix* C){
	int size = A->n;
	if(size % 16 != 0){
		std::cerr << "wrong matrix size, aborting\n";
		return;
	}
	// reordering B for using cache & 
	// prioritizing columns over rows
	for(int j = 0; j < size; j+=16){
		//submatrix size_c*16 to sequence for caching
		buf16 B_buf(size);
		b_reorder_to_16(&B_buf, B->m + j, size);
		for(int i = 0; i < size; i+=4){
			init_res_core(4,C->m + i*size + j, size);
			microcore_4x16(size, 
				A->m + i*size, B_buf.d, 16, 
				C->m + i*size + j);
		}
	}
}