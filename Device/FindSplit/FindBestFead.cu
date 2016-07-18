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
#include "../DeviceHashing.h"
#include "../prefix-sum/prefixSum.h"
#include "../svm-shared/devUtility.h"
#include "../../DeviceHost/NodeStat.h"


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
								   const float_point *pfLocalBestGain, const int *pnLocalBestGainKey,
								   float_point *pfFeaGlobalBestGain, int *pnFeaGlobalBestGainKey,
								   int nLocalBlockPerFeaInBatch)
{
	//blockIdx.x (==1) corresponds to a feature which has multiple values
	//blockIdx.y corresponds to a feature id
	//blockIdx.z corresponds to a splittable node id
	int nGlobalThreadId = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

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
								   const float_point *pfFeaGlobalBestGain, const int *pnFeaGlobalBestGainKey,
								   float_point *pfBlockBestFea, int *pnBlockBestKey)
{
	//blockIdx.y corresponds to a splittable node id

	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int snId = blockIdx.y;
	if(snId >= numofSNode)
		printf("Error in PickBestFea: kernel split %d nods, but only %d splittable nodes\n", snId, numofSNode);

	__shared__ float_point pfGain[BLOCK_SIZE];
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
								   const float_point *pfBlockBestFea, const int *pnBlockBestFeaKey,
								   float_point *pfGlobalBestFea, int *pnGlobalBestKey)
{
	int localTId = threadIdx.x;
	int snId = blockIdx.x;//a block per splittable node
	if(snId >= numofSNode)
		printf("numof block is larger than the numof splittable nodes! %d v.s. %d \n", snId, numofSNode);

	__shared__ float_point pfGain[BLOCK_SIZE];
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


__global__ void FindSplitInfo(const int *pnKeyValue, const long long *plFeaStartPos, const float_point *pFeaValue,
							  int feaBatch, int smallestFeaId,
							  const float_point *pfGlobalBestFea, const int *pnGlobalBestKey, const int *pBuffId,
							  const nodeStat *snNodeStat, const float_point *pPrefixSumGD, const float_point *pPrefixSumHess,
							  SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat,
							  float_point *pLastValue, const float_point *pGainOnEachFeaValue_d)
{

	//get current fea batch size
/*	long long startPosOfSmallest = plFeaStartPos[smallestFeaId];
	int largestFeaId = smallestFeaId + feaBatch - 1;
	long long startPosOfLargest = plFeaStartPos[largestFeaId];
	int batchSize = startPosOfLargest - startPosOfSmallest + pnKeyValue[largestFeaId];
*/	int batchSize;
	int tempFirstFeaStartPos;//it is for satisfying the function call
	GetBatchInfo(feaBatch, smallestFeaId, 0, pnKeyValue, plFeaStartPos, tempFirstFeaStartPos, batchSize);

	int snId = threadIdx.x;
	int key = pnGlobalBestKey[snId];//position in a batch for one node.
	int buffId = pBuffId[snId];

	//compute feature id
	int bestFeaId = -1;
	int valuePos = -1;
	for(int f = smallestFeaId; f < feaBatch + smallestFeaId; f++)
	{
		int numofKeyValue = pnKeyValue[f];
		if(plFeaStartPos[f] + numofKeyValue < key)
			continue;
		else
		{
			bestFeaId = f;
			valuePos = key - plFeaStartPos[f];
			break;
		}
	}

	pBestSplitPoint[buffId].m_fGain = -pfGlobalBestFea[snId];//change the gain back to positive
//	if(pBestSplitPoint[buffId].m_fGain > 9.69 && pBestSplitPoint[buffId].m_fGain < 9.7  && bestFeaId == 5)
//		printf("Here have a look\n");
	if(-pfGlobalBestFea[snId] <= 0)//no gain
	{
		return;
	}
	pBestSplitPoint[buffId].m_nFeatureId = bestFeaId;
	int svPos = plFeaStartPos[bestFeaId] + valuePos;
	if(svPos <= 0)
		printf("Error in FindSplitInfo: split point is at %d!\n", svPos);
	pBestSplitPoint[buffId].m_fSplitValue = 0.5f * (pFeaValue[svPos] + pFeaValue[svPos - 1]);

	pLastValue[buffId] = pFeaValue[svPos];

	//child node stat
	int posInWholeBatch = batchSize * snId + key - 1;
	if(posInWholeBatch < 0)
		printf("posInWholeBatch is negative: %d!\n", posInWholeBatch);

	pLChildStat[buffId].sum_gd = snNodeStat[buffId].sum_gd - pPrefixSumGD[posInWholeBatch];
	pLChildStat[buffId].sum_hess = snNodeStat[buffId].sum_hess - pPrefixSumHess[posInWholeBatch];
	pRChildStat[buffId].sum_gd = pPrefixSumGD[posInWholeBatch];
	pRChildStat[buffId].sum_hess = pPrefixSumHess[posInWholeBatch];
	if(pLChildStat[buffId].sum_hess < 0 || pRChildStat[buffId].sum_hess < 0)
		printf("Error: hess is negative l hess=%d, r hess=%d, svPos=%d\n", pLChildStat[buffId].sum_hess, pRChildStat[buffId].sum_hess, svPos);
}
