/*
 * DeviceSplitterKernel.cu
 *
 *  Created on: 10 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>

#include "DeviceFindFeaKernel.h"
#include "DeviceSplitter.h"
#include "../KernelConst.h"
#include "../DeviceHashing.h"
#include "../svm-shared/devUtility.h"
#include <float.h>

const float rt_2eps = 2.0 * DeviceSplitter::rt_eps;

__global__ void FindFeaSplitValue(const int *pnNumofKeyValues, const long long *pnFeaStartPos, const int *pInsId, const float_point *pFeaValue,
								  const int *pInsIdToNodeId, const float_point *pGD, const float_point *pHess,
								  nodeStat *pTempRChildStatPerThread, float_point *pLastValuePerThread,
								  const nodeStat *pSNodeStatPerThread, SplitPoint *pBestSplitPointPerThread,
								  nodeStat *pRChildStatPerThread, nodeStat *pLChildStatPerThread,
								  const int *pSNIdToBuffId, int maxNumofSplittable, const int *pBuffId, int numofSNode,
								  float_point lambda, int numofFea)
{
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	int feaId = nGlobalThreadId;
	if(feaId >= numofFea)
	{
		return;
	}

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
	long long startPosOfCurFea = startPosOfPrevFea + numofPreFeaKeyValues;
	const int *InsIdStartAddress = pInsId + startPosOfCurFea;
	const float_point *pInsValueStartAddress = pFeaValue + startPosOfCurFea;

    for(int i = 0; i < pnNumofKeyValues[nGlobalThreadId]; i++)
    {
    	int insId = InsIdStartAddress[i];
    	int nid = pInsIdToNodeId[insId];
		if(nid < -1)
		{
			printf("Error: nid=%d\n", nid);
			return;
		}
		if(nid == -1)
			continue;

		// start working
		double fvalue = pInsValueStartAddress[i];

		// get the buffer id of node nid
		int hashValue = GetBufferId(pSNIdToBuffId, nid, maxNumofSplittable);
		int bufferPos = hashValue + feaId * maxNumofSplittable;

		if(pTempRChildStatPerThread[bufferPos].sum_hess == 0.0)//equivalent to IsEmpty()
		{
			pTempRChildStatPerThread[bufferPos].sum_gd += pGD[insId];
			pTempRChildStatPerThread[bufferPos].sum_hess += pHess[insId];
			pLastValuePerThread[bufferPos] = fvalue;
		}
		else
		{
			// try to find a split
			if(fabs(fvalue - pLastValuePerThread[bufferPos]) > rt_2eps)
			{
				//SNodeStatPerThread is the same for all the features, so using hashValue is fine and can save memory
				float_point tempGD = pSNodeStatPerThread[hashValue].sum_gd - pTempRChildStatPerThread[bufferPos].sum_gd;
				float_point tempHess = pSNodeStatPerThread[hashValue].sum_hess - pTempRChildStatPerThread[bufferPos].sum_hess;
				bool needUpdate = NeedUpdate(pTempRChildStatPerThread[bufferPos].sum_hess, tempHess);
				if(needUpdate == true)
				{
					double sv = (fvalue + pLastValuePerThread[bufferPos]) * 0.5f;
					if(hashValue == 1)
					{
//						double loss_chg = CalGain(pSNodeStatPerThread[bufferPos], pTempRChildStatPerThread[bufferPos], tempGD, tempHess, lambda);
//						printf("nid=%d, sv=%f, gain=%f\n", nid, sv, loss_chg);
					}

		            UpdateSplitInfo(pSNodeStatPerThread[hashValue], pBestSplitPointPerThread[bufferPos], pRChildStatPerThread[bufferPos],
		            							  pLChildStatPerThread[bufferPos], pTempRChildStatPerThread[bufferPos], tempGD, tempHess,
		            							  lambda, sv, feaId);
				}
			}
			//update the statistics
			pTempRChildStatPerThread[bufferPos].sum_gd += pGD[insId];
			pTempRChildStatPerThread[bufferPos].sum_hess += pHess[insId];
			pLastValuePerThread[bufferPos] = fvalue;
		}
	}


    // finish updating all statistics, check if it is possible to include all sum statistics
    for(int i = 0; i < numofSNode; i++)
    {
    	if(pBuffId[i] < 0)
    		printf("Error in buffer id %d, i=%d, numofSN=%d\n", pBuffId[i], i, numofSNode);

    	int hashVaue = pBuffId[i];
    	int buffId = hashVaue + feaId * maxNumofSplittable;//an id in the buffer
    	float_point tempGD = pSNodeStatPerThread[hashVaue].sum_gd - pTempRChildStatPerThread[buffId].sum_gd;
    	float_point tempHess = pSNodeStatPerThread[hashVaue].sum_hess - pTempRChildStatPerThread[buffId].sum_hess;
    	bool needUpdate = NeedUpdate(pTempRChildStatPerThread[buffId].sum_hess, tempHess);
        if(needUpdate == true)
        {
            const float delta = fabs(pLastValuePerThread[buffId]) + DeviceSplitter::rt_eps;
            float_point sv = pLastValuePerThread[buffId] + delta;

            UpdateSplitInfo(pSNodeStatPerThread[hashVaue], pBestSplitPointPerThread[buffId], pRChildStatPerThread[buffId], pLChildStatPerThread[buffId],
            							  pTempRChildStatPerThread[buffId], tempGD, tempHess, lambda, sv, feaId);
        }
    }
}

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

__device__ double CalGain(const nodeStat &parent, const nodeStat &r_child,
						  const float_point &l_child_GD, const float_point &l_child_Hess,
						  const float_point &lambda)
{
	PROCESS_ERROR(abs(parent.sum_gd - l_child_GD - r_child.sum_gd) < 0.0001);
	PROCESS_ERROR(parent.sum_hess == l_child_Hess + r_child.sum_hess);

//	printf("lgd=%f, lhe=%f, rgd=%f, rhe=%f, pgd=%f, phe=%f, lamb=%f\n", l_child_GD, l_child_Hess,
//			r_child.sum_gd, r_child.sum_hess, parent.sum_gd, parent.sum_hess, lambda);

	//compute the gain
	double fGain = (l_child_GD * l_child_GD)/(l_child_Hess + lambda) +
				   (r_child.sum_gd * r_child.sum_gd)/(r_child.sum_hess + lambda) -
				   (parent.sum_gd * parent.sum_gd)/(parent.sum_hess + lambda);
//	if(fGain > -10)
//	{
//		printf("gain=%f, lgd=%f, lhe=%f, rgd=%f, rhe=%f, pgd=%f, phe=%f, lamb=%f\n", fGain, l_child_GD, l_child_Hess,
//				r_child.sum_gd, r_child.sum_hess, parent.sum_gd, parent.sum_hess, lambda);
//	}


	return fGain;
}


 __device__ bool UpdateSplitPoint(SplitPoint &curBest, double fGain, double fSplitValue, int nFeatureId)
{
	if(fGain > curBest.m_fGain )//|| (fGain == m_fGain && nFeatureId == m_nFeatureId) NOT USE (second condition is for updating to a new split value)
	{
		curBest.m_fGain = fGain;
		curBest.m_fSplitValue = fSplitValue;
		curBest.m_nFeatureId = nFeatureId;
		return true;
	}
	return false;
}

__device__ void UpdateLRStat(nodeStat &RChildStat, nodeStat &LChildStat,
							 const nodeStat &TempRChildStat, const float_point &grad, const float_point &hess)
{
	LChildStat.sum_gd = grad;
	LChildStat.sum_hess = hess;
	RChildStat = TempRChildStat;
}

__device__ bool NeedUpdate(float_point &RChildHess, float_point &LChildHess)
{
	if(LChildHess >= DeviceSplitter::min_child_weight && RChildHess >= DeviceSplitter::min_child_weight)
		return true;
	return false;
}

__device__ void UpdateSplitInfo(const nodeStat &snStat, SplitPoint &bestSP, nodeStat &RChildStat, nodeStat &LChildStat,
								const nodeStat &TempRChildStat, const float_point &tempGD, const float_point &tempHess,
								const float_point &lambda, const float_point &sv, const int &featureId)
{
	double loss_chg = CalGain(snStat, TempRChildStat, tempGD, tempHess, lambda);
    bool bUpdated = UpdateSplitPoint(bestSP, loss_chg, sv, featureId);
	if(bUpdated == true)
	{
		UpdateLRStat(RChildStat, LChildStat, TempRChildStat, tempGD, tempHess);
	}
}

