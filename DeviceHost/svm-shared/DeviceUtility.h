/**
 * devUtility.h
 * @brief: This file contains InitCUDA() function and a reducer class CReducer
 * Created on: May 24, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef SVM_DEVUTILITY_H_
#define SVM_DEVUTILITY_H_
//include files from the gpu sdk
#include <cuda_runtime.h>
#include "../../SharedUtility/DataType.h"
#include "../../SharedUtility/CudaMacro.h"

/**
 * @brief: when the array is too large to store in shared memory, one thread loads multiple values
 */
template<class T>
__device__ void GetGlobalMinPreprocessing(int nArraySize, const real *pfBlockMinValue, const T *pnBlockMinKey,
										  real *pfSharedMinValue, T *pnSharedMinKey)
{
	int localTid = threadIdx.x;
	if(nArraySize > BLOCK_SIZE)
	{
		if(BLOCK_SIZE != blockDim.x)
			printf("Error: Block size inconsistent in PickFeaGlobalBestSplit\n");

		real fTempMin = pfSharedMinValue[localTid];
		T nTempMinKey = pnSharedMinKey[localTid];
		for(int i = localTid + BLOCK_SIZE; i < nArraySize; i += blockDim.x)
		{
			real fTempBlockMin = pfBlockMinValue[i];
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
template<class T>
__device__ void LoadToSharedMem(int nArraySize, int gainStartPos,
								const real *pfBlockMinValue, const T *pnBlockMinKey,
		  	  	  	  	  	    real *pfSharedMinValue, T *pnSharedMinKey)
{
	int localTId = threadIdx.x;
	int firstElementPos = gainStartPos + localTId;
	pfSharedMinValue[localTId] = pfBlockMinValue[firstElementPos];
	pnSharedMinKey[localTId] = pnBlockMinKey[firstElementPos];

	//if the size of block is larger than the BLOCK_SIZE, we make the size to be not larger than BLOCK_SIZE
	//the thread loads more elements
	GetGlobalMinPreprocessing(nArraySize, pfBlockMinValue + gainStartPos, pnBlockMinKey + gainStartPos, pfSharedMinValue, pnSharedMinKey);
}


#endif /* SVM_DEVUTILITY_H_ */
