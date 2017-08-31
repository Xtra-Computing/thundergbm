#include <stdio.h>
#include "getMin.h"


__device__ int getBlockMin(const float *values, int *index) {
	CONCHECKER(blockDim.x % 32 == 0);
	int tid = threadIdx.x;
	index[tid] = tid;
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (tid < offset) {
			if (values[index[tid + offset]] < values[index[tid]]) {
				index[tid] = index[tid + offset];
			}
		}
		__syncthreads();
	}
	return index[0];
}

__device__ void GetMinValueOriginal(real *pfValues)
{
	CONCHECKER(blockDim.x % 32 == 0);
	//Reduce by a factor of 2, and minimize step size
	for (int i = blockDim.x / 2; i > 0 ; i >>= 1) {
		int tid = threadIdx.x;
		if (tid < i)
			if (pfValues[tid + i] < pfValues[tid])
				pfValues[tid] = pfValues[tid + i];
        __syncthreads();
	}
}
