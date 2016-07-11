/*
 * FindBestFead.cu
 *
 *  Created on: 8 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: find best feature with its split point using massive threads
 */

#include <float.h>
#include <stdio.h>
#include "FindFeaKernel.h"
#include "../KernelConst.h"
#include "../prefix-sum/prefixSum.h"
#include "../svm-shared/devUtility.h"
#include "../../DeviceHost/NodeStat.h"

__device__ void GetBatchInfo(int feaBatch, int smallestFeaId, int feaId, const int *pnNumofKeyValues, const long long *pnFeaStartPos,
							 int &curFeaStartPosInBatch, int &nFeaValueInBatch)
{
	int largetFeaId = smallestFeaId + feaBatch - 1;
	int smallestFeaIdStartPos = pnFeaStartPos[smallestFeaId];//first feature start pos of this batch
	int largestFeadIdStartPos = pnFeaStartPos[largetFeaId];   //last feature start pos of this batch
	curFeaStartPosInBatch = pnFeaStartPos[feaId] - smallestFeaIdStartPos;
	nFeaValueInBatch = largestFeadIdStartPos - smallestFeaIdStartPos + pnNumofKeyValues[largetFeaId];
}

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void ObtainGDEachNode(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const int *pInsId, const float_point *pFeaValue,
		  const int *pInsIdToNodeId, const float_point *pGD, const float_point *pHess,  int numofSNode, int smallestFeaId, int totalNumofFea, int feaBatch,
		  float_point *pGDOnEachFeaValue, float_point *pHessOnEachFeaValue, float_point *pValueOneEachFeaValue)
{
	//blockIdx.x corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id
	//## global id looks ok, but need to be careful
	int nGlobalThreadId = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	int snId = blockIdx.z;
	if(snId >= numofSNode)
		printf("# of block groups is larger than # of splittable nodes: %d v.s. %d\n", snId, numofSNode);

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
	}
	else
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
									  int totalNumofFea, int feaBatch, int *pStartPosEachFeaInBatch, int *pnEachFeaLen)
{
	int feaId = blockIdx.x * blockDim.x + threadIdx.x + smallestFeaId;//add a shift here to process only part of the features
	int feaIdInBatch = blockIdx.x * blockDim.x + threadIdx.x;
	if(feaId >= totalNumofFea)
		printf("Error in GetStartPosEachFeaInBatch: feaId=%d, while total numofFea=%d\n", feaId, totalNumofFea);

	int smallestFeaIdStartPos = pnFeaStartPos[smallestFeaId];//first feature start pos of this batch
	int curFeaStartPosInBatch = pnFeaStartPos[feaId] - smallestFeaIdStartPos;
	pStartPosEachFeaInBatch[feaIdInBatch] = curFeaStartPosInBatch;
	pnEachFeaLen[feaIdInBatch] = pnNumofKeyValues[feaId];
}

/**
 * @brief: compute the prefix sum for gd and hess
 */
void PrefixSumForEachNode(int feaBatch, float_point *pGDOnEachFeaValue_d, float_point *pHessOnEachFeaValue_d,
						  const int *pnStartPosEachFeaInBatch, const int *pnEachFeaLen)
{
	prefixsumForDeviceArray(pGDOnEachFeaValue_d, pnStartPosEachFeaInBatch, pnEachFeaLen, feaBatch);
	prefixsumForDeviceArray(pHessOnEachFeaValue_d, pnStartPosEachFeaInBatch, pnEachFeaLen, feaBatch);
}

/**
 * @brief: compute the gain in parallel, each gain is computed by a thread; kernel have the same configuration as obtainGDEachNode.
 */
__global__ void ComputeGain(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const nodeStat *pSNodeStat,
							int smallestFeaId, int feaBatch, const int *pBuffId, int numofSNode, float_point lambda,
							const float_point *pGDPrefixSumOnEachFeaValue, const float_point *pHessPrefixSumOnEachFeaValue,
							float_point *pGainOnEachFeaValue)
{
	//blockIdx.x corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id
	int nGlobalThreadId = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	int snId = blockIdx.z;
	int hashVaue = pBuffId[snId];
	if(hashVaue < 0)
		printf("Error in ComputeGain: buffer id %d, i=%d, numofSN=%d\n", hashVaue, snId, numofSNode);

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
	float_point rChildGD = pGDPrefixSumOnEachFeaValue[bufferPos];
	float_point rChildHess = pHessPrefixSumOnEachFeaValue[bufferPos];
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
 * @brief: pick best feature of this batch for all the splittable nodes
 * Each block.y processes one node, a thread processes a reduction.
 */
__global__ void PickFeaLocalBestSplit(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const float_point *pGainOnEachFeaValue,
								   const int *pBuffId, int smallestFeaId, int feaBatch,
								   int numofSNode, int maxNumofSplittable,
								   float_point *pfBestGain, int *pnBestGainKey)
{
	//blockIdx.x corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id
	int nGlobalThreadId = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	int snId = blockIdx.z;
	int feaId = blockIdx.y + smallestFeaId;
	if(blockIdx.y >= feaBatch)
		printf("the feaBatch is smaller than blocks for feas: %d v.s. %d\n", feaBatch, blockIdx.y);

	if(snId >= numofSNode)
		printf("Error in PickBestFea: kernel split %d nods, but only %d splittable nodes\n", snId, numofSNode);
	if(pBuffId[snId] < 0 || pBuffId[snId] >= maxNumofSplittable)
		printf("Error in PickBestFea\n");

	__shared__ float_point pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[localTid] = -1;

	int numofValueOfThisFea = pnNumofKeyValues[feaId];//get the number of key-value pairs of this feature
	int tidForEachFeaValue = blockIdx.x * blockDim.x + threadIdx.x;
	if(blockIdx.x * blockDim.x >= numofValueOfThisFea)
	{
		printf("###### this is a dummy block, need to fill the block here\n");
		//may be "pfBestGain[blockId] = -FLT_MAX;"
	}

	if(tidForEachFeaValue >= numofValueOfThisFea)
	{
		return;
	}


	//load gain and entry id to shared memory
	if(pBuffId[snId] < 0)
		printf("pBuffId[snId] < 0! is %d\n", pBuffId[snId]);

	int curFeaStartPosInBatch;
	int nFeaValueInBatch;
	GetBatchInfo(feaBatch, smallestFeaId, feaId, pnNumofKeyValues, pnFeaStartPos, curFeaStartPosInBatch, nFeaValueInBatch);
	int nPos = snId * nFeaValueInBatch + curFeaStartPosInBatch + tidForEachFeaValue;//compute splittable node position in buffer
	if(nPos < 0)
		printf("sp pos is nagative! %d\n", nPos);

	pfGain[localTid] = -pGainOnEachFeaValue[nPos];//change to find min of -gain
	pnBetterGainKey[localTid] = nPos;
	__syncthreads();

	//find the local best split point
	GetMinValue(pfGain, pnBetterGainKey, blockDim.x);
	__syncthreads();
	int blockId = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
	if(localTid == 0)//copy the best gain to global memory
	{
		pfBestGain[blockId] = pfGain[0];
		pnBestGainKey[blockId] = pnBetterGainKey[0];
		if(pnBetterGainKey[0] < 0)
			printf("negative key: snId=%d, feaId=%d, blockId=%d, gain=%f, key=%d\n", snId, feaId, blockId, pfGain[0], pnBetterGainKey[0]);
	}
}


/**
 * @brief: pick best feature of this batch for all the splittable nodes
 * Each block.y processes one node, a thread processes a reduction.
 */
__global__ void PickFeaGlobalBestSplit(int feaBatch, int numofSNode,
								   const float_point *pfLocalBestGain, const int *pnLocalBestGainKey,
								   float_point *pfFeaGlobalBestGain, int *pnFeaGlobalBestGainKey,
								   int nLocalBlockOfPerFeaInBatch)
{
	//blockIdx.x corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id
	int nGlobalThreadId = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	int snId = blockIdx.z;
	if(blockIdx.y >= feaBatch)
		printf("the feaBatch is smaller than blocks for feas: %d v.s. %d\n", feaBatch, blockIdx.y);

	if(snId >= numofSNode)
		printf("Error in PickBestFea: kernel split %d nods, but only %d splittable nodes\n", snId, numofSNode);
	if(blockIdx.x > 0 || gridDim.x != 1)
		printf("#### one block is not enough to find global best split for a feature!\n");

	__shared__ float_point pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[localTid] = -1;

	int posInFeaLocalBest = threadIdx.x;

	if(posInFeaLocalBest >= nLocalBlockOfPerFeaInBatch)//number of threads is larger than the number of blocks
	{
		return;
	}

	int posOfLocalBest = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	pfGain[localTid] = pfLocalBestGain[posOfLocalBest];//change to find min of -gain
	pnBetterGainKey[localTid] = pnLocalBestGainKey[posOfLocalBest];

	//if the size of block is larger than the BLOCK_SIZE, we make the size to be not larger than BLOCK_SIZE
	const float_point *pfGainStartPos = pfLocalBestGain + posOfLocalBest;
	const int *pnGainKeyStartPos = pnLocalBestGainKey + posOfLocalBest;
	GetGlobalMinPreprocessing(nLocalBlockOfPerFeaInBatch, pfGainStartPos, pnGainKeyStartPos, pfGain, pnBetterGainKey);
	 __syncthreads();	//wait until the thread within the block

	//find the local best split point
	GetMinValue(pfGain, pnBetterGainKey, blockDim.x);
	__syncthreads();
	int feaBlockId = blockIdx.z * gridDim.y + blockIdx.y;//the best split for a feature
	if(localTid == 0)//copy the best gain to global memory
	{
		pfFeaGlobalBestGain[feaBlockId] = pfGain[0];
		pnFeaGlobalBestGainKey[feaBlockId] = pnBetterGainKey[0];
		if(pnBetterGainKey[0] < 0)
			printf("negative key: snId=%d, feaBlockId=%d, gain=%f, key=%d\n", snId, feaBlockId, pfGain[0], pnBetterGainKey[0]);
	}
}

