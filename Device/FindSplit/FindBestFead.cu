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
#include "../DeviceHashing.h"
#include "../../DeviceHost/svm-shared/DeviceUtility.h"
#include "../../DeviceHost/NodeStat.h"
#include "../../SharedUtility/CudaMacro.h"


/**
 * @brief: pick best feature of this batch for all the splittable nodes
 * Each block.y processes one node, a thread processes a reduction.
 */
__global__ void PickFeaLocalBestSplit(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const real *pGainOnEachFeaValue,
								   const int *pBuffId, int smallestFeaId, int feaBatch,
								   int numofSNInProgress, int smallestNodeId, int maxNumofSplittable,
								   real *pfBestGain, int *pnBestGainKey)
{
	//blockIdx.x corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id

	int snId = blockIdx.z;
	int feaId = blockIdx.y + smallestFeaId;
	if(blockIdx.y >= feaBatch)
		printf("the feaBatch is smaller than blocks for feas: %d v.s. %d\n", feaBatch, blockIdx.y);

	if(snId >= numofSNInProgress)
		printf("Error in PickBestFea: kernel split %d nods, but only %d splittable nodes\n", snId, numofSNInProgress);
	if(pBuffId[snId + smallestNodeId] < 0 || pBuffId[snId + smallestNodeId] >= maxNumofSplittable)
		printf("Error in PickBestFea\n");

	__shared__ real pfGain[BLOCK_SIZE];
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
	if(pBuffId[snId + smallestNodeId] < 0)
		printf("pBuffId[snId] < 0! is %d\n", pBuffId[snId + smallestNodeId]);

	int curFeaStartPosInBatch;
	int nFeaValueInBatch;
	GetBatchInfo(feaBatch, smallestFeaId, feaId, pnNumofKeyValues, pnFeaStartPos, curFeaStartPosInBatch, nFeaValueInBatch);
	int nPos = snId * nFeaValueInBatch + curFeaStartPosInBatch + tidForEachFeaValue;//compute splittable node position in buffer
	if(nPos < 0)
		printf("sp pos is nagative! %d\n", nPos);

	pfGain[localTid] = -pGainOnEachFeaValue[nPos];//change to find min of -gain
	pnBetterGainKey[localTid] = curFeaStartPosInBatch + tidForEachFeaValue;
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
 */
__global__ void PickFeaGlobalBestSplit(int feaBatch, int numofSNode,
								   const real *pfLocalBestGain, const int *pnLocalBestGainKey,
								   real *pfFeaGlobalBestGain, int *pnFeaGlobalBestGainKey,
								   int nLocalBlockPerFeaInBatch)
{
	//blockIdx.x (==1) corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id

	int snId = blockIdx.z;
	if(blockIdx.y >= feaBatch)
		printf("the feaBatch is smaller than blocks for feas: %d v.s. %d\n", feaBatch, blockIdx.y);

	if(snId >= numofSNode)
		printf("Error in PickBestFea: kernel split %d nods, but only %d splittable nodes\n", snId, numofSNode);
	if(blockIdx.x > 0 || gridDim.x != 1)
		printf("#### one block is not enough to find global best split for a feature!\n");

	__shared__ real pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[localTid] = -1;

	int posInFeaLocalBest = threadIdx.x;

	if(posInFeaLocalBest >= nLocalBlockPerFeaInBatch)//number of threads is larger than the number of blocks
	{
		return;
	}

	int feaIdInBatch = blockIdx.y;
	int curFeaLocalBestStartPos = (snId * feaBatch + feaIdInBatch) * nLocalBlockPerFeaInBatch;

	LoadToSharedMem(nLocalBlockPerFeaInBatch, curFeaLocalBestStartPos, pfLocalBestGain, pnLocalBestGainKey, pfGain, pnBetterGainKey);
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

/**
 * @brief: pick best feature of this batch for all the splittable nodes
 */
__global__ void PickLocalBestFeaBestSplit(int feaBatch, int numofSNode,
								   const real *pfFeaGlobalBestGain, const int *pnFeaGlobalBestGainKey,
								   real *pfBlockBestFea, int *pnBlockBestKey)
{
	//blockIdx.y corresponds to a splittable node id

	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int snId = blockIdx.y;
	if(snId >= numofSNode)
		printf("Error in PickBestFea: kernel split %d nods, but only %d splittable nodes\n", snId, numofSNode);

	__shared__ real pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initalise to a large positive number
	pnBetterGainKey[localTid] = -1;

	int feaIdInBatch = blockIdx.x * blockDim.x + threadIdx.x;;
	if(feaIdInBatch >= feaBatch)
	{//the number of threads in the block is larger than the number of blocks
		return;
	}

	int fFeaBestGainPos = feaIdInBatch + snId * feaBatch;
	pfGain[localTid] = pfFeaGlobalBestGain[fFeaBestGainPos];
	pnBetterGainKey[localTid] = pnFeaGlobalBestGainKey[fFeaBestGainPos];
	__syncthreads();

	//find the local best split point
	GetMinValue(pfGain, pnBetterGainKey, blockDim.x);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		pfBlockBestFea[blockId] = pfGain[0];
		pnBlockBestKey[blockId] = pnBetterGainKey[0];
		if(pnBetterGainKey[0] < 0)
			printf("the key should never be negative!\n");
	}
}

__global__ void PickGlobalBestFeaBestSplit(int numofSNode, int nBlockPerNode,
								   const real *pfBlockBestFea, const int *pnBlockBestFeaKey,
								   real *pfGlobalBestFea, int *pnGlobalBestKey)
{
	int localTId = threadIdx.x;
	int snId = blockIdx.x;//a block per splittable node
	if(snId >= numofSNode)
		printf("numof block is larger than the numof splittable nodes! %d v.s. %d \n", snId, numofSNode);

	__shared__ real pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	pfGain[localTId] = FLT_MAX;
	pnBetterGainKey[localTId] = -1;
	if(localTId >= nBlockPerNode)
	{
		printf("block size is %d v.s. numof blocks per node is %d\n", blockDim.x, nBlockPerNode);
		printf("Error: local thread id %d larger than numof blocks %d per node. They should be the same.\n", localTId, nBlockPerNode);
		return;
	}
	int gainStartPos = snId * nBlockPerNode;

	LoadToSharedMem(nBlockPerNode, gainStartPos, pfBlockBestFea, pnBlockBestFeaKey, pfGain, pnBetterGainKey);
	__syncthreads();

	//find the local best split point
	GetMinValue(pfGain, pnBetterGainKey, nBlockPerNode);
	__syncthreads();
	if(localTId == 0)//copy the best gain to global memory
	{
		pfGlobalBestFea[snId] = pfGain[0];
		pnGlobalBestKey[snId] = pnBetterGainKey[0];
	}
}

