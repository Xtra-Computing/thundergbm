/*
 * getMinPreprocessing.cu
 *
 *  Created on: 11 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "DeviceUtility.h"
#include "../../SharedUtility/CudaMacro.h"

/**
 * @brief: when the array is too large to store in shared memory, one thread loads multiple values
 */
__device__ void GetGlobalMinPreprocessing(int nArraySize, const real *pfBlockMinValue, const int *pnBlockMinKey,
										  real *pfSharedMinValue, int *pnSharedMinKey)
{
	int localTid = threadIdx.x;
	if(nArraySize > BLOCK_SIZE)
	{
		if(BLOCK_SIZE != blockDim.x)
			printf("Error: Block size inconsistent in PickFeaGlobalBestSplit\n");

		real fTempMin = pfSharedMinValue[localTid];
		int nTempMinKey = pnSharedMinKey[localTid];
		for(int i = localTid + BLOCK_SIZE; i < nArraySize; i += blockDim.x)
		{
			real fTempBlockMin = pfBlockMinValue[i];
			if(fTempBlockMin < fTempMin)
			{
				if(pnBlockMinKey[i] < 0)
					printf("negative key is included into the final result! tid=%d\n", localTid);
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
								const real *pfBlockMinValue, const int *pnBlockMinKey,
		  	  	  	  	  	    real *pfSharedMinValue, int *pnSharedMinKey)
{
	int localTId = threadIdx.x;
	int firstElementPos = gainStartPos + localTId;
	pfSharedMinValue[localTId] = pfBlockMinValue[firstElementPos];
	pnSharedMinKey[localTId] = pnBlockMinKey[firstElementPos];

	//if the size of block is larger than the BLOCK_SIZE, we make the size to be not larger than BLOCK_SIZE
	//the thread loads more elements
	GetGlobalMinPreprocessing(nArraySize, pfBlockMinValue + gainStartPos, pnBlockMinKey + gainStartPos, pfSharedMinValue, pnSharedMinKey);
}


