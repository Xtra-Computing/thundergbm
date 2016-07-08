/*
 * prefixSum.cu
 *
 *  Created on: 6 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "prefixSum.h"

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

#define NUM_BLOCKS 511
#define BLOCK_SIZE 512


/**
 * @brief: compute prefix sum for in_array which contains multiple arrays of similar length
 */
__global__ void cuda_prefixsum(T *in_array, T *out_array, int *arrayStartPos, int numofBlock, unsigned int *pnThreadLastBlock, unsigned int *pnEltsLastBlock)
{
	// shared should be sized to blockDim.x
	extern __shared__ T shared[];

	unsigned int array_id = blockIdx.y;	//blockIdx.y corresponds to an array to compute prefix sum
	int numEltsLastBlock = pnEltsLastBlock[array_id];
	int numThreadLastBlock = pnThreadLastBlock[array_id];
	unsigned int array_len = (numofBlock - 1) * blockDim.x + numEltsLastBlock;
	unsigned int tid = threadIdx.x;
	unsigned int b_offset = arrayStartPos[array_id] + blockIdx.x * blockDim.x;//in_array offset due to multi blocks
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
__global__ void cuda_updatesum(T *array, int *arrayStartPos, T *update_array)
{
	extern __shared__ T shared[];

	unsigned int tid = threadIdx.x;
	int array_id = blockIdx.y;
	unsigned int b_offset = arrayStartPos[array_id] + blockIdx.x * blockDim.x;//in_array offset due to multi blocks
	unsigned int id = b_offset + threadIdx.x;
	int op = 0;

	if (blockIdx.x > 0)//if it is not the first block
	{
		op = update_array[blockIdx.y * gridDim.x + blockIdx.x - 1];
	}

	shared[tid] = array[id] + op;
	array[id] = shared[tid];
}

