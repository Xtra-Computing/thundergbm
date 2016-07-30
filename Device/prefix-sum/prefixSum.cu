/*
 * prefixSum.cu
 *
 *  Created on: 6 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "prefixSum.h"

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define testing true
/**
 * @brief: compute prefix sum for in_array which contains multiple arrays of similar length
 */
__global__ void cuda_prefixsum(T *in_array, int in_array_size, T *out_array, const long long *arrayStartPos, const unsigned int *pnEachSubArrayLen,
							   int numArray, int numofBlockPerSubArray, unsigned int *pnThreadLastBlock, unsigned int *pnEltsLastBlock)
{
	// shared should be sized to blockDim.x
	extern __shared__ T shared[];

	int blockIdAllArray = (blockIdx.z * gridDim.y + blockIdx.y);//blockIdAllArray corresponds to an array to compute prefix sum
	if(blockIdAllArray >= numArray)
		return;
	unsigned int array_id = blockIdAllArray;
	int numEltsLastBlock = pnEltsLastBlock[array_id];
	int numThreadLastBlock = pnThreadLastBlock[array_id];
	unsigned int array_len = pnEachSubArrayLen[array_id];
	unsigned int tid = threadIdx.x;
	//####### long long to unsigned int here
	unsigned int b_offset = arrayStartPos[array_id] + blockIdx.x * blockDim.x * 2;//in_array offset due to multi blocks; each thread handles two elements
	if(blockIdx.x * blockDim.x * 2 >= array_len)//this block is dummy, due to the small subarray size
	{
		out_array[blockIdAllArray * gridDim.x + blockIdx.x] = 0;
		return;
	}
	//check if it is the last block
	bool isLastBlock = false;
	if((blockIdx.x + 1) * blockDim.x * 2 >= array_len)//next block is invalid
		isLastBlock = true;

	unsigned int offset = 1;

	int numElementInBlock;
	if(isLastBlock == false)//not the last block
		numElementInBlock = blockDim.x * 2;
	else//the last block
		numElementInBlock = numThreadLastBlock * 2;//last block may process more (or fewer) elements than the rest.

	if(true == isLastBlock && (tid >= numThreadLastBlock))//skip threads with large indices
		return;

	int i = tid;
	int j = tid + numElementInBlock / 2;

	int offset_i = CONFLICT_FREE_OFFSET(i);
	int offset_j = CONFLICT_FREE_OFFSET(j);

#if testing
	if(i + offset_i >= blockDim.x * 2)
		printf("index is out of bound of shared memory: i=%d, offset_i=%d\n", i, offset_i);
	if(i + b_offset >= in_array_size)
	{
		printf("index is out of bound of array: i=%d, b_offset=%d, array size=%d\n", i, b_offset, in_array_size);
	}
#endif

	shared[i + offset_i] = in_array[i + b_offset];
	if(true == isLastBlock && j >= numEltsLastBlock)//the last block
		shared[j + offset_j] = 0;
	else
	{
#if testing
		if(j + offset_j >= blockDim.x * 2)
			printf("index is out of bound of shared memory: j=%d, offset_j=%d\n", j, offset_j);
		if(j + b_offset >= in_array_size)
		{
			printf("index is out of bound of array: lastBlock=%d, aid=%d, j=%d, b_offset=%d, array size=%d, numEltsLastBlock=%d\n",
					isLastBlock, array_id, j, b_offset, in_array_size, numEltsLastBlock);
		}
#endif
		shared[j + offset_j] = in_array[j + b_offset];
	}

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
			if(true == isLastBlock && lastElementPos >= numEltsLastBlock)//last block
			{
				lastElementPos = numEltsLastBlock - 1;
			}

			in_array[b_offset + lastElementPos] += shared[lastElementPos + CONFLICT_FREE_OFFSET(lastElementPos)];
			out_array[blockIdAllArray * gridDim.x + blockIdx.x] = in_array[b_offset + lastElementPos];//block sum
		}

		__syncthreads();
		if (false == isLastBlock)//not last block
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
__global__ void cuda_updatesum(T *array, const long long *arrayStartPos, const unsigned int *pnEachSubArrayLen, int numArray, T *update_array)
{
	int blockIdAllArray = (blockIdx.z * gridDim.y + blockIdx.y);//blockIdAllArray corresponds to an array to compute prefix sum
	if(blockIdAllArray >= numArray)
		return;
	unsigned int tid = threadIdx.x;
	int array_id = blockIdAllArray;
	//##### long long to unsigned int here
	unsigned int b_offset = arrayStartPos[array_id] + blockIdx.x * blockDim.x;//in_array offset due to multi blocks
	unsigned int id = b_offset + tid;
	unsigned int array_len = pnEachSubArrayLen[array_id];
	if(blockIdx.x * blockDim.x + tid >= array_len)//skip this thread, due to the small subarray size
	{
		return;
	}

	T op = 0;
	if (blockIdx.x > 0)//if it is not the first block
	{
		op = update_array[blockIdAllArray * gridDim.x + blockIdx.x - 1];
	}

	T temp = array[id] + op;
	array[id] = temp;
}

