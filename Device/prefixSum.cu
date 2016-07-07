/*
 * prefixSum.cu
 *
 *  Created on: 6 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>

#include "prefixSum.h"

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define NUM_BLOCKS 511
#define BLOCK_SIZE 512

using std::cout;
using std::endl;


/**
 * @brief: compute prefix sum for in_array with the number of elements equals to "size".
 * @size: size should be power of two.
 */
__global__ void cuda_prefixsum(T *in_array, T *out_array, int numofBlock, int numThreadLastBlock, int numEltsLastBlock)
{
	// shared should be sized to blockDim.x
	extern __shared__ T shared[];

	unsigned int tid = threadIdx.x;
	unsigned int b_offset = blockIdx.x * blockDim.x;//in_array offset due to multi blocks
	unsigned int offset = 1;

	int numElementInBlock;
	if(blockIdx.x < numofBlock - 1)//not the last block
		numElementInBlock = blockDim.x;
	else//the last block
		numElementInBlock = numThreadLastBlock * 2;

	if((blockIdx.x == numofBlock - 1) && (tid >= numThreadLastBlock))//skip threads with large indices
		return;

	int i = tid;
	int j = tid + numElementInBlock / 2;

	int offset_i = CONFLICT_FREE_OFFSET(i);
	int offset_j = CONFLICT_FREE_OFFSET(j);

	shared[i + offset_i] = in_array[i + b_offset];
	if((blockIdx.x == numofBlock - 1) && j >= numEltsLastBlock)
		shared[j + offset_j] = 0;
	else
		shared[j + offset_j] = in_array[j + b_offset];

	// scan up
	for (int s = (numElementInBlock >> 1); s > 0; s >>= 1)
	{
		__syncthreads();

		if (tid < s) {
			int i = offset * (2 * tid + 1) - 1;
			int j = offset * (2 * tid + 2) - 1;
			i += CONFLICT_FREE_OFFSET(i);
			j += CONFLICT_FREE_OFFSET(j);
			shared[j] += shared[i];
		}
		offset <<= 1;
	}

	if (tid == 0)
	{
		shared[numElementInBlock - 1 + CONFLICT_FREE_OFFSET(numElementInBlock - 1)] = 0;
	}
	// scan down
	for (int s = 1; s < numElementInBlock; s <<= 1)
	{
		offset >>= 1;
		__syncthreads();

		if (tid < s)
		{
			int i = offset * (2 * tid + 1) - 1;
			int j = offset * (2 * tid + 2) - 1;
			i += CONFLICT_FREE_OFFSET(i);
			j += CONFLICT_FREE_OFFSET(j);
			T tmp = shared[i];
			shared[i] = shared[j];
			shared[j] += tmp;
		}
	}
	__syncthreads();
	// copy data back to main memory
	// scan is exclusive, make it inclusive by left shifting elements
	if (tid < numElementInBlock / 2)
	{
		if (tid > 0)
		{
			in_array[b_offset + i - 1] = shared[i + offset_i];
		}
		else//tid == 0
		{
			// re-calc the last element, drop it in out array
			int lastElementPos = numElementInBlock - 1;
			if((blockIdx.x == numofBlock - 1) && lastElementPos >= numEltsLastBlock)//last block
			{
				lastElementPos = numEltsLastBlock - 1;
			}

			in_array[b_offset + lastElementPos] += shared[lastElementPos + CONFLICT_FREE_OFFSET(lastElementPos)];
			out_array[blockIdx.x] = in_array[b_offset + lastElementPos];//block sum
		}

		if (blockIdx.x < numofBlock - 1)//not last block
			in_array[b_offset + j - 1] = shared[j + offset_j];
		else//last block
		{
			 if(j < numEltsLastBlock)
			 {
				 in_array[b_offset + j - 1] = shared[j + offset_j];
			 }
		}
	}
}


/**
 * @brief: post processing of prefix sum for large array
 */
__global__ void cuda_updatesum(T *array, T *update_array, int size)
{
	extern __shared__ T shared[];

	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	int op = 0;

	if (blockIdx.x > 0)
	{
		op = update_array[blockIdx.x - 1];
	}

	shared[tid] = array[id] + op;
	array[id] = shared[tid];
}

inline bool isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127);
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

/**
 * @brief: prefix sum for an array in device memory
 */
void prefixsumForDeviceArray(T *array_d, int size)
{
	if(isPowerOfTwo(size) == false)
	{
		cout << "array size is not power of two" << endl;
	}

	int numElements = size;
    unsigned int blockSize = 64; // max size of the thread blocks
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;//one thread processes two elements.

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;//only one block and
    else
        numThreads = floorPow2(numElements);//a few threads only have one element to process.

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadLastBlock;//threads for the last block
    if (isPowerOfTwo(numEltsLastBlock))
    	numThreadLastBlock = numEltsLastBlock / 2;
    else
        numThreadLastBlock = floorPow2(numEltsLastBlock);

	T *out_array_d;
	T *tmp_d;

	dim3 dim_grid(numBlocks, 1, 1);
	dim3 dim_block(numThreads, 1, 1);

	// allocate temp, block sum, and device arrays
	cudaMalloc((void **)&tmp_d, numBlocks * sizeof(T));
	cudaMalloc((void **)&out_array_d, numBlocks * sizeof(T));

	// do prefix sum for each block
	cuda_prefixsum <<< dim_grid, dim_block, numEltsPerBlock * sizeof(T) >>> (array_d, out_array_d, numBlocks, numThreadLastBlock, numEltsLastBlock);

	// do prefix sum for block sum
    unsigned int numThreadBlockSum;//threads for the last block
    if (isPowerOfTwo(numBlocks))
    	numThreadBlockSum = numBlocks / 2;
    else
    	numThreadBlockSum = floorPow2(numBlocks);
    unsigned int numBlockForBlockSum = 1;
	cuda_prefixsum <<< dim_grid, dim_block, numEltsPerBlock * sizeof(T) >>> (out_array_d, tmp_d, numBlockForBlockSum, numThreadBlockSum, numBlocks);
	// update original array using block sum
	cuda_updatesum <<< dim_grid, dim_block, numEltsPerBlock * sizeof(T) >>> (array_d, out_array_d, size);

	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in prefixsumForDeviceArray" << endl;
		exit(0);
	}

	cudaFree(out_array_d);
	cudaFree(tmp_d);
}

/**
 * @brief: prefix sum for an array in host memory
 */
void prefixsumForHostArray(int blocks, int threads, T *array_h, int size)
{
	T *array_d;

	// allocate temp, block sum, and device arrays
	cudaMalloc((void **)&array_d, size * sizeof(T));
	cudaMemcpy(array_d, array_h, size * sizeof(T), cudaMemcpyHostToDevice);

	prefixsumForDeviceArray(array_d, size);

	// copy resulting array back to host
	cudaMemcpy(array_h, array_d, size * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(array_d);
}

///////////////// for testing
void prefixsum_host(T *array_h, int size)
{
	for (int i = 0; i < size; i++) {
		if (i > 0) {
			array_h[i] += array_h[i - 1];
		}
	}
}

void usage(int which)
{
	switch (which) {
	default:
		printf("usage: prefixsum [-h|-b blocks|-t threads] max\n");
		break;
	case 1:
		printf("prefixsum requires numbers <= threads*blocks\n");
		break;
	}
}

void print_array(T *array, int count)
{
	for (int i = 0; i < count; i++) {
		cout << array[i] << endl;
	}
}

void prepare_numbers(T **array, int count)
{
	T *numbers = new T[count];

	// load array
	for (int i = 0; i < count; i++) {
		numbers[i] = i + 1.0;
	}

	*array = numbers;
}



int TestPrefixSum(int argc, char *argv[])
{
	int opt, host_mode, blocks, threads, max;
	T *array;

	// set options
	host_mode = 0;
	blocks = 1;
	threads = 64;
	while ((opt = getopt(argc, argv, "hd")) != -1) {
		switch (opt) {
		case 'h':
			host_mode = 1;
			break;
		case 'd':
			host_mode = 0;
			break;
		default:
			usage(0);
			return 0;
		}
	}

	// check to make sure we are feeding in correct number of args
	if (argc == optind + 1) {
		max = atoi(argv[optind]);
	} else {
		usage(0);
		return 0;
	}
	// pre-init numbers
	array = NULL;
	prepare_numbers(&array, max);

	if (host_mode) {
		printf("prefix sum using host\n");
		prefixsum_host(array, max);
	} else {
		printf("prefix sum using CUDA\n");
		prefixsumForHostArray(blocks, threads, array, max);
	}

	// print array
	print_array(array, max);

	free(array);

	return 0;
}
