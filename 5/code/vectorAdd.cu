#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"


void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void printVector(int * v, size_t n);


int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	const int vectorSize = 1024;
	int a[vectorSize], b[vectorSize], c[vectorSize];
	
	fillVector(a, vectorSize);
	fillVector(b, vectorSize);
	
	cudaEventRecord(start);
	addVector(a, b, c, vectorSize);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	printVector(c, vectorSize);
	
	float elapsed_time = 0;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("elapsed_time : %f", elapsed_time);
	
	return EXIT_SUCCESS;
}

// Fills a vector with data
void fillVector(int * v, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		v[i] = i;
	}
}

// Adds two vectors
void addVector(int * a, int *b, int *c, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

// Prints a vector to the stdout.
void printVector(int * v, size_t n) {
	int i;
	printf("[-] Vector elements: ");
	for (i = 0; i < n; i++) {
		printf("%d, ", v[i]);
	}
	printf("\b\b  \n");
}
