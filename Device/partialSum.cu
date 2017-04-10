/*
 * partialSum.cu
 *
 *  Created on: 1 Aug 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "partialSum.h"

__global__ void blockSum(float_point * input, float_point * output, int len, bool isFinalSum)
{
	int blockSize = blockDim.x;
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    //@@ Load a segment of the input vector into shared memory
    extern __shared__ float_point partialSum[];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockId * blockSize;//each thread adds two values

    //thread loads first element
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;

    //thread loads second element
    if (start + blockSize + t < len)
       partialSum[blockSize + t] = input[start + blockSize + t];
    else
       partialSum[blockSize + t] = 0;

    if(isFinalSum == true)//handle the last block
    {
    	for(int i = 2 * blockSize + t; i < len; i += blockSize)
    	{
    		partialSum[t] += input[start + i];
    	}
    }

    //@@ Traverse the reduction tree
    for (unsigned int stride = blockSize; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0)
       output[blockId] = partialSum[0];
}

