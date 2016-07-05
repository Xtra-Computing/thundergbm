/*
 * PickBestSplit.cu
 *
 *  Created on: 5 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: pick the best feature with a split value among a set of features with their split points.
 */

#include <float.h>
#include <stdio.h>
#include "FindFeaKernel.h"
#include "../KernelConst.h"
#include "../svm-shared/devUtility.h"

__device__ void CopyNodeStat(nodeStat *pDest, const nodeStat *pSrc)
{
	pDest[0].sum_gd = pSrc[0].sum_gd;
	pDest[0].sum_hess = pSrc[0].sum_hess;
}

/**
 * @brief: pick best feature for all the splittable nodes
 * Each block.y processes one node, a thread processes a reduction.
 */
__global__ void PickLocalBestFea(const SplitPoint *pBestSplitPointPerThread, const int *pBuffId,
								 int numofSNode, int numofFea, int maxNumofSplittable,
								 float_point *pfBestGain, int *pnBestGainKey)
{
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int snId = blockIdx.y;
	if(snId >= numofSNode)
		printf("Error in PickBestFea: kernel split %d nods, but only %d splittable nodes\n", snId, numofSNode);
	if(pBuffId[snId] < 0 || pBuffId[snId] >= maxNumofSplittable)
		printf("Error in PickBestFea\n");

	__shared__ float_point pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initalise to a large positive number
	pnBetterGainKey[localTid] = -1;

	int feaId = blockIdx.x * blockDim.x + threadIdx.x;;
	if(feaId >= numofFea)
	{
		return;
	}

	//load gain and entry id to shared memory
	if(pBuffId[snId] < 0)
		printf("pBuffId[snId] < 0! is %d\n", pBuffId[snId]);
	int nSplitPointPos = feaId * maxNumofSplittable + pBuffId[snId];//compute splittable node position in buffer
	if(nSplitPointPos < 0)
		printf("sp pos is nagative! %d\n", nSplitPointPos);
	if(nSplitPointPos > 19998 * 100 + pBuffId[snId])
		printf("should not happen!\n");

	if(pBestSplitPointPerThread[nSplitPointPos].m_nFeatureId != feaId &&
	   pBestSplitPointPerThread[nSplitPointPos].m_nFeatureId != -1)//some feature may not have a good split point.
		printf("diff between fea id in best split and real fea id: %d v.s. %d\n", pBestSplitPointPerThread[nSplitPointPos].m_nFeatureId, feaId);

	pfGain[localTid] = -pBestSplitPointPerThread[nSplitPointPos].m_fGain;//change to find min
	pnBetterGainKey[localTid] = nSplitPointPos;
	__syncthreads();

	//find the local best split point
	GetMinValue(pfGain, pnBetterGainKey, blockDim.x);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		pfBestGain[blockId] = pfGain[0];
		pnBestGainKey[blockId] = pnBetterGainKey[0];
		if(pnBetterGainKey[0] < 0)
			printf("the key should never be negative!\n");
	}
}

/**
 * @brief: find the best split point
 * use one block for each node
 */
__global__ void PickGlobalBestFea(float_point *pLastValuePerThread,
							SplitPoint *pBestSplitPointPerThread, nodeStat *pRChildStatPerThread, nodeStat *pLChildStatPerThread,
							const int *pBuffId, int numofSNode,
							const float_point *pfBestGain, const int *pnBestGainKey, int numofBlockPerNode)
{
	int localTId = threadIdx.x;
	int snId = blockIdx.x;
	if(snId >= numofSNode)
		printf("numof block is larger than the numof splittable nodes! %d v.s. %d \n", snId, numofSNode);
	int firstElementPos = snId * numofBlockPerNode + threadIdx.x;

	__shared__ float_point pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	pfGain[localTId] = pfBestGain[firstElementPos];
	pnBetterGainKey[localTId] = pnBestGainKey[firstElementPos];
	if(localTId >= numofBlockPerNode)
	{
		printf("block size is %d v.s. numof blocks per node is %d\n", blockDim.x, numofBlockPerNode);
		printf("Error: local thread id %d larger than numof blocks %d per node. They should be the same.\n", localTId, numofBlockPerNode);
		return;
	}

	int gainStartPos = snId * numofBlockPerNode;
	for(int i = firstElementPos; i < gainStartPos + numofBlockPerNode; i += blockDim.x)//for handling the number of blocks larger than the block size
	{
		if(pfGain[localTId] > pfBestGain[i])//we change to find the smallest value by previously negating the gain.
		{
			pfGain[localTId] = pfBestGain[i];
			if(pnBestGainKey[i] < 0)
				printf("key=%d, i=%d, globalTid=%d, blockId=%d\n", pnBestGainKey[i], i, firstElementPos, blockIdx.x);
			pnBetterGainKey[localTId] = pnBestGainKey[i];
		}
	}
	__syncthreads();
	//find the local best split point
	GetMinValue(pfGain, pnBetterGainKey, numofBlockPerNode);
	if(localTId == 0)//copy the best gain to global memory
	{
		int nBestGainKey = pnBetterGainKey[0];
		int nodePos = pBuffId[snId];//compute splittable node position in buffer

		pLastValuePerThread[nodePos] = pLastValuePerThread[nBestGainKey];

		pBestSplitPointPerThread[nodePos].m_fGain = pBestSplitPointPerThread[nBestGainKey].m_fGain;
		pBestSplitPointPerThread[nodePos].m_fSplitValue = pBestSplitPointPerThread[nBestGainKey].m_fSplitValue;
		pBestSplitPointPerThread[nodePos].m_nFeatureId = pBestSplitPointPerThread[nBestGainKey].m_nFeatureId;

		CopyNodeStat(pRChildStatPerThread + nodePos, pRChildStatPerThread + nBestGainKey);
		CopyNodeStat(pLChildStatPerThread + nodePos, pLChildStatPerThread + nBestGainKey);
	}
}


