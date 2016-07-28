/*
 * getMinPreprocessing.cu
 *
 *  Created on: 11 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "DeviceUtility.h"
#include "../../Device/KernelConst.h"

/**
 * @brief: when the array is too large to store in shared memory, one thread loads multiple values
 */
__device__ void GetGlobalMinPreprocessing(int nArraySize, const float_point *pfBlockMinValue, const int *pnBlockMinKey,
										  float_point *pfSharedMinValue, int *pnSharedMinKey)
{
	int localTid = threadIdx.x;
	if(nArraySize > BLOCK_SIZE)
	{
		if(BLOCK_SIZE != blockDim.x)
			printf("Error: Block size inconsistent in PickFeaGlobalBestSplit\n");

		float_point fTempMin = pfSharedMinValue[localTid];
		int nTempMinKey = pnSharedMinKey[localTid];
		for(int i = localTid + BLOCK_SIZE; i < nArraySize; i += blockDim.x)
		{
			float_point fTempBlockMin = pfBlockMinValue[i];
			if(fTempBlockMin < fTempMin)
			{
			//store the minimum value and the corresponding key
				fTempMin = fTempBlockMin;
				nTempMinKey = pnBlockMinKey[i];
			}
		}
		pnSharedMinKey[localTid] = nTempMinKey;
		pfSharedMinValue[localTid] = fTempMin;
	}
}

/**
 * @brief: load to shared memory
 */
__device__ void LoadToSharedMem(int nArraySize, int gainStartPos,
								const float_point *pfBlockMinValue, const int *pnBlockMinKey,
		  	  	  	  	  	    float_point *pfSharedMinValue, int *pnSharedMinKey)
{
	int localTId = threadIdx.x;
	int firstElementPos = gainStartPos + localTId;
	pfSharedMinValue[localTId] = pfBlockMinValue[firstElementPos];
	pnSharedMinKey[localTId] = pnBlockMinKey[firstElementPos];

	//if the size of block is larger than the BLOCK_SIZE, we make the size to be not larger than BLOCK_SIZE
	//the thread loads more elements
	GetGlobalMinPreprocessing(nArraySize, pfBlockMinValue + gainStartPos, pnBlockMinKey + gainStartPos, pfSharedMinValue, pnSharedMinKey);
}


