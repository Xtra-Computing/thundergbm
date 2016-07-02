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
#include "../DeviceHashing.h"

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
	if(feaId > numofFea)
	{
		printf("should not happened!\n");
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
				float_point tempGD = pSNodeStatPerThread[bufferPos].sum_gd - pTempRChildStatPerThread[bufferPos].sum_gd;
				float_point tempHess = pSNodeStatPerThread[bufferPos].sum_hess - pTempRChildStatPerThread[bufferPos].sum_hess;
				bool needUpdate = NeedUpdate(pTempRChildStatPerThread[bufferPos].sum_hess, tempHess);
				if(needUpdate == true)
				{
					double sv = (fvalue + pLastValuePerThread[bufferPos]) * 0.5f;
					if(hashValue == 1)
					{
//						double loss_chg = CalGain(pSNodeStatPerThread[bufferPos], pTempRChildStatPerThread[bufferPos], tempGD, tempHess, lambda);
//						printf("nid=%d, sv=%f, gain=%f\n", nid, sv, loss_chg);
					}

		            UpdateSplitInfo(pSNodeStatPerThread[bufferPos], pBestSplitPointPerThread[bufferPos], pRChildStatPerThread[bufferPos],
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

    	int buffId = pBuffId[i] + feaId * maxNumofSplittable;//an id in the buffer
    	float_point tempGD = pSNodeStatPerThread[buffId].sum_gd - pTempRChildStatPerThread[buffId].sum_gd;
    	float_point tempHess = pSNodeStatPerThread[buffId].sum_hess - pTempRChildStatPerThread[buffId].sum_hess;
    	bool needUpdate = NeedUpdate(pTempRChildStatPerThread[buffId].sum_hess, tempHess);
        if(needUpdate == true)
        {
            const float delta = fabs(pLastValuePerThread[buffId]) + DeviceSplitter::rt_eps;
            float_point sv = pLastValuePerThread[buffId] + delta;

            UpdateSplitInfo(pSNodeStatPerThread[buffId], pBestSplitPointPerThread[buffId], pRChildStatPerThread[buffId], pLChildStatPerThread[buffId],
            							  pTempRChildStatPerThread[buffId], tempGD, tempHess, lambda, sv, feaId);
        }
    }
}

__device__ void CopyNodeStat(nodeStat *pDest, const nodeStat *pSrc)
{
	pDest[0].sum_gd = pSrc[0].sum_gd;
	pDest[0].sum_hess = pSrc[0].sum_hess;
}

__global__ void PickBestFea(float_point *pLastValuePerThread,
							SplitPoint *pBestSplitPointPerThread, nodeStat *pRChildStatPerThread, nodeStat *pLChildStatPerThread,
							const int *pBuffId, int numofSNode, int numofFea, int maxNumofSplittable)
{
	//the best splittable node is stored in the first numofSNode nodes.
	for(int f = 0; f < numofFea; f++)
	{//Caution when to do in parallel
		for(int n = 0; n < numofSNode; n++)
		{
			if(pBuffId[n] < 0 || pBuffId[n] >= maxNumofSplittable)
				printf("Error in PickBestFea\n");
			int nodePos = f * maxNumofSplittable + pBuffId[n];//compute splittable node position in buffer
			int firstPosOfSplitPoint = pBuffId[n];
			if(pBestSplitPointPerThread[nodePos].m_fGain > pBestSplitPointPerThread[firstPosOfSplitPoint].m_fGain)
			{
				pLastValuePerThread[firstPosOfSplitPoint] = pLastValuePerThread[nodePos];

				pBestSplitPointPerThread[firstPosOfSplitPoint].m_fGain = pBestSplitPointPerThread[nodePos].m_fGain;
				pBestSplitPointPerThread[firstPosOfSplitPoint].m_fSplitValue = pBestSplitPointPerThread[nodePos].m_fSplitValue;
				pBestSplitPointPerThread[firstPosOfSplitPoint].m_nFeatureId = pBestSplitPointPerThread[nodePos].m_nFeatureId;

				CopyNodeStat(pRChildStatPerThread + firstPosOfSplitPoint, pRChildStatPerThread + nodePos);
				CopyNodeStat(pLChildStatPerThread + firstPosOfSplitPoint, pLChildStatPerThread + nodePos);
			}
		}
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

