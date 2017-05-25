/*
 * FindFeaUtility.cu
 *
 *  Created on: 14 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "FindFeaKernel.h"
#include "../Splitter/DeviceSplitter.h"

__device__ void GetBatchInfo(int feaBatch, int smallestFeaId, int feaId, const int *pnNumofKeyValues, const long long *pnFeaStartPos,
							 int &curFeaStartPosInBatch, int &nFeaValueInBatch)
{
	int largetFeaId = smallestFeaId + feaBatch - 1;
	int smallestFeaIdStartPos = pnFeaStartPos[smallestFeaId];//first feature start pos of this batch
	int largestFeadIdStartPos = pnFeaStartPos[largetFeaId];   //last feature start pos of this batch
	curFeaStartPosInBatch = pnFeaStartPos[feaId] - smallestFeaIdStartPos;
	nFeaValueInBatch = largestFeadIdStartPos - smallestFeaIdStartPos + pnNumofKeyValues[largetFeaId];
}


__device__ bool NeedUpdate(real &RChildHess, real &LChildHess)
{
	if(LChildHess >= DeviceSplitter::min_child_weight && RChildHess >= DeviceSplitter::min_child_weight)
		return true;
	return false;
}
