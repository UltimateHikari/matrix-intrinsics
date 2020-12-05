#pragma once
#include <iostream>
#include <fstream>
#include <immintrin.h>

using namespace std;

struct Matrix{
	int n;
	float* m; //matrix
	Matrix(int n_): n(n_){
		m = new float[n*n];
	}
	Matrix(int n_, bool p): Matrix(n_){
		// I
		for(int i = 0; i < n; ++i){
			m[i*n + i] = 1.0;
		}
	}

	~Matrix(){delete [] m;}

	void out(){
		ofstream fout("output.txt", std::ios::app);
		for(int i = 0; i< n; ++i){
			for(int j = 0; j < n; ++j){
				fout << m[i*n + j] << ' ';
			}
			fout << endl;
		}
		fout.close();
	}
};

struct buf16{
	float* d;
	int n;
	buf16(int size):
		n(size),
		d((float*)_mm_malloc(16 * size * 4, 64))
		{}
	~buf16(){
		_mm_free(d);
	}
};

void trivial_sum(Matrix* A, Matrix* B);
void trivial_sub(Matrix* A, Matrix* B);
void trivial_mult_opt(Matrix* A, Matrix* B, Matrix* C);

void vector_mult(Matrix* A, Matrix* B, Matrix* C);
void vector_mult_opt(Matrix* A, Matrix* B, Matrix* C);