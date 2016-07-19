/*
 * ComputeGain.cu
 *
 *  Created on: 14 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "FindFeaKernel.h"
#include "../KernelConst.h"
#include "../DeviceHashing.h"
#include "../prefix-sum/prefixSum.h"
#include "../Splitter/DeviceSplitter.h"

const float rt_2eps = 2.0 * DeviceSplitter::rt_eps;

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void ObtainGDEachNode(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const int *pInsId, const float_point *pFeaValue,
		  const int *pInsIdToNodeId, const float_point *pGD, const float_point *pHess, const int *pBuffId, const int *pSNIdToBuffId,
		  int maxNumofSplittable, int numofSNodeInProgress, int smallestNodeId, int smallestFeaId, int totalNumofFea, int feaBatch,
		  float_point *pGDOnEachFeaValue, float_point *pHessOnEachFeaValue, float_point *pValueOneEachFeaValue)
{
	//blockIdx.x corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id
	//## global id looks ok, but need to be careful
	int nGlobalThreadId = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	int snId = blockIdx.z;
	if(snId >= numofSNodeInProgress)
		printf("# of block groups is larger than # of splittable nodes: %d v.s. %d\n", snId, numofSNodeInProgress);

	int snHashValue = pBuffId[snId + smallestNodeId];//hash value (buffer position) of the splittable node
	if(snHashValue < 0)
		printf("node id is incorrect resulting in hash valule of %d\n", snHashValue);
	int feaId = blockIdx.y + smallestFeaId;//add a shift here to process only part of the features

	int curFeaStartPosInBatch;
	int nFeaValueInBatch;
	GetBatchInfo(feaBatch, smallestFeaId, feaId, pnNumofKeyValues, pnFeaStartPos, curFeaStartPosInBatch, nFeaValueInBatch);

	int numofValueOfThisFea = pnNumofKeyValues[feaId];
	if(feaId >= totalNumofFea)
		printf("# of block groups is larger than # of features: %d v.s. %d\n", feaId, totalNumofFea);

	//addresses of instance ids and key-value pairs
		//compute start position key-value pairs of the current feature
	long long startPosOfPrevFea = 0;
	int numofPreFeaKeyValues = 0;

	if(feaId > 0)
	{
		//number of key values of the previous feature
		numofPreFeaKeyValues = pnNumofKeyValues[feaId - 1];
		//copy value of the start position of the previous feature
		startPosOfPrevFea = pnFeaStartPos[feaId - 1];
	}
	int tidForEachFeaValue = blockIdx.x * blockDim.x + threadIdx.x;
	if(tidForEachFeaValue >= numofValueOfThisFea)
	{
		return;
	}

	long long startPosOfCurFea = startPosOfPrevFea + numofPreFeaKeyValues;
	const int *InsIdStartAddress = pInsId + startPosOfCurFea;
	const float_point *pInsValueStartAddress = pFeaValue + startPosOfCurFea;

	int insId = InsIdStartAddress[tidForEachFeaValue];
	int nid = pInsIdToNodeId[insId];
	if(nid < -1)
	{
		printf("Error: nid=%d\n", nid);
		return;
	}
	if(nid == -1)
	{//some leave nodes
		return;
	}

	int hashValue = GetBufferId(pSNIdToBuffId, nid, maxNumofSplittable);
	if(snHashValue != hashValue)//the instance does not belong to this splittable node
	{
		//set GD/Hess to 0 in this position. Since the default value is 0, no action is required.
		return;
	}

	{//some splittable nodes
		int bufferPos = snId * nFeaValueInBatch + curFeaStartPosInBatch + tidForEachFeaValue;
		if(pGDOnEachFeaValue[bufferPos] != 0 || pHessOnEachFeaValue[bufferPos]!= 0 || pValueOneEachFeaValue[bufferPos] != 0)
			printf("default value of gd/hess/fvalue is incorrect in ObtainGDEachNode. snId=%d\n", snId);

		//GD/Hess of the same node is stored consecutively.
		pGDOnEachFeaValue[bufferPos] = pGD[insId];
		pHessOnEachFeaValue[bufferPos] = pHess[insId];
		pValueOneEachFeaValue[bufferPos] = pInsValueStartAddress[tidForEachFeaValue];
	}
}

/**
 * @brief: each thread computes the start position in the batch for each feature
 */
__global__ void GetInfoEachFeaInBatch(const int *pnNumofKeyValues, const long long *pnFeaStartPos, int smallestFeaId,
									  int totalNumofFea, int feaBatch, int numofSNInProgress, int smallestNodeId,
									  int *pStartPosEachFeaInBatch, int *pnEachFeaLen)
{
	int feaId = blockIdx.x * blockDim.x + threadIdx.x + smallestFeaId;//add a shift here to process only part of the features
	int feaIdInBatch = blockIdx.x * blockDim.x + threadIdx.x;
	if(feaId >= totalNumofFea)
	{
//		printf("Error in GetStartPosEachFeaInBatch: feaId=%d, while total numofFea=%d\n", feaId, totalNumofFea);
		return;
	}

	int oneNodeBatchSize;
	int curFeaStartPosInBatch;
	GetBatchInfo(feaBatch, smallestFeaId, feaId, pnNumofKeyValues, pnFeaStartPos, curFeaStartPosInBatch, oneNodeBatchSize);

	for(int i = 0; i < numofSNInProgress; i++)
	{
		pStartPosEachFeaInBatch[feaIdInBatch + i * feaBatch] = oneNodeBatchSize * i + curFeaStartPosInBatch;
		pnEachFeaLen[feaIdInBatch + i * feaBatch] = pnNumofKeyValues[feaId];
	}
}

/**
 * @brief: compute the prefix sum for gd and hess
 */
void PrefixSumForEachNode(int numofSubArray, float_point *pGDOnEachFeaValue_d, float_point *pHessOnEachFeaValue_d,
						  const int *pnStartPosEachFeaInBatch, const int *pnEachFeaLen)
{
	prefixsumForDeviceArray(pGDOnEachFeaValue_d, pnStartPosEachFeaInBatch, pnEachFeaLen, numofSubArray);
	prefixsumForDeviceArray(pHessOnEachFeaValue_d, pnStartPosEachFeaInBatch, pnEachFeaLen, numofSubArray);
}

/**
 * @brief: compute the gain in parallel, each gain is computed by a thread; kernel have the same configuration as obtainGDEachNode.
 */
__global__ void ComputeGain(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const nodeStat *pSNodeStat,
							int smallestFeaId, int feaBatch, const int *pBuffId,
							int numofSNInProgress, int smallestNodeId, float_point lambda,
							const float_point *pGDPrefixSumOnEachFeaValue, const float_point *pHessPrefixSumOnEachFeaValue,
							const float_point *pFeaValue,
							float_point *pGainOnEachFeaValue)
{
	//blockIdx.x corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id
	int nGlobalThreadId = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	int snId = blockIdx.z;
	int hashVaue = pBuffId[snId + smallestNodeId];
	if(hashVaue < 0)
		printf("Error in ComputeGain: buffer id %d, i=%d, numofSN=%d\n", hashVaue, snId, numofSNInProgress);

	int feaId = blockIdx.y + smallestFeaId;//add a shift here to process only part of the features
	int numofValueOfThisFea = pnNumofKeyValues[feaId];
	int tidForEachFeaValue = blockIdx.x * blockDim.x + threadIdx.x;
	if(tidForEachFeaValue >= numofValueOfThisFea)
	{
		return;
	}

	int curFeaStartPosInBatch;
	int nFeaValueInBatch;
	GetBatchInfo(feaBatch, smallestFeaId, feaId, pnNumofKeyValues, pnFeaStartPos, curFeaStartPosInBatch, nFeaValueInBatch);

	int bufferPos = snId * nFeaValueInBatch + curFeaStartPosInBatch + tidForEachFeaValue;
	if(tidForEachFeaValue == 0)
	{
		//assign gain to 0
    	pGainOnEachFeaValue[bufferPos] = 0;
		return;
	}

	int exclusiveSumPos = bufferPos - 1;//following xgboost using exclusive sum on gd and hess
	if(exclusiveSumPos < 0)
		printf("Index to get prefix sum is negative: %d\n", exclusiveSumPos);
	float_point rChildGD = pGDPrefixSumOnEachFeaValue[exclusiveSumPos];
	float_point rChildHess = pHessPrefixSumOnEachFeaValue[exclusiveSumPos];
	float_point snGD = pSNodeStat[hashVaue].sum_gd;
	float_point snHess = pSNodeStat[hashVaue].sum_hess;
	float_point tempGD = snGD - rChildGD;
	float_point tempHess = snHess - rChildHess;
	bool needUpdate = NeedUpdate(rChildHess, tempHess);
    if(needUpdate == true)//need to compute the gain
    {
    	pGainOnEachFeaValue[bufferPos] = (tempGD * tempGD)/(tempHess + lambda) +
    									 (rChildGD * rChildGD)/(rChildHess + lambda) - (snGD * snGD)/(snHess + lambda);
    }
    else
    {
    	//assign gain to 0
    	pGainOnEachFeaValue[bufferPos] = 0;
    }

}

/**
 * @brief: compute the gain in parallel, each gain is computed by a thread; kernel have the same configuration as obtainGDEachNode.
 */
__global__ void FixedGain(const int *pnNumofKeyValues, const long long *pnFeaStartPos,
						  int smallestFeaId, int feaBatch, int numofSNInProgress, int smallestNodeId,
						  const float_point *pHessOnEachFeaValue, const float_point *pFeaValue,
						  float_point *pGainOnEachFeaValue, float_point *pLastBiggerValue)
{
	//blockIdx.x corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id
	int nGlobalThreadId = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int snId = blockIdx.z;
	if(snId >= numofSNInProgress)
		printf("The number of super blocks is larger than the number of splittable nodes.\n");

	int feaId = blockIdx.y + smallestFeaId;//add a shift here to process only part of the features
	int numofValueOfThisFea = pnNumofKeyValues[feaId];
	int tidForEachFeaValue = blockIdx.x * blockDim.x + threadIdx.x;
	if(tidForEachFeaValue >= numofValueOfThisFea)
	{
		return;
	}

	int curFeaStartPosInBatch;
	int nFeaValueInBatch;
	GetBatchInfo(feaBatch, smallestFeaId, feaId, pnNumofKeyValues, pnFeaStartPos, curFeaStartPosInBatch, nFeaValueInBatch);

	int elePos = curFeaStartPosInBatch + tidForEachFeaValue;
	int bufferPos = snId * nFeaValueInBatch + elePos;

	if(pGainOnEachFeaValue[bufferPos] == 0)//gain is zero; don't need to fix
		return;

	float_point fvalue = pFeaValue[elePos];
	int previousElePos = elePos - 1;
	int previousBufferPos = bufferPos - 1;
    while(previousElePos >= curFeaStartPosInBatch)//try if we can erase this gain
    {
    	if(fabs(pFeaValue[previousElePos] - fvalue) > rt_2eps)
    	{
    		pLastBiggerValue[bufferPos] = pFeaValue[previousElePos];//Note: some last bigger values are not accurate for improving efficiency!
    		break;
    	}
    	else
    	{
    		if(pHessOnEachFeaValue[previousBufferPos] == 1)
    		{//hessian value of previous element with same feaValue is 1 meaning the current gain is invalid
    			pGainOnEachFeaValue[bufferPos] = 0;
    			break;
    		}
    		else
    		{
    			previousElePos--;
    			previousBufferPos--;
    		}
    	}
    }
}
