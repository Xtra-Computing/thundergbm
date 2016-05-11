/*
 * DeviceSplitterKernel.cu
 *
 *  Created on: 10 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>

#include "DeviceSplitterKernel.h"
#include "DeviceSplitter.h"

const float rt_2eps = 2.0 * DeviceSplitter::rt_eps;

__global__ void FindFeaSplitValue(int nNumofKeyValues, int *idStartAddress, float_point *pValueStartAddress, int *pInsIdToNodeId,
								  nodeStat *pTempRChildStat, float_point *pGD, float_point *pHess, float_point *pLastValue,
								  nodeStat *pSNodeState, SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat,
								  int *pSNIdToBuffId, int maxNumofSplittable, int featureId, int *pBuffId, int numofSNode, float_point lambda)
{
    for(int i = 0; i < nNumofKeyValues; i++)
    {
    	int insId = idStartAddress[i];
    	int nid = pInsIdToNodeId[insId];
		if(nid < -1)
		{
			printf("Error: nid=%d\n", nid);
			return;
		}
		if(nid == -1)
			continue;

		// start working
		double fvalue = pValueStartAddress[i];

		// get the buffer id of node nid
		int bufferPos = GetBufferId(pSNIdToBuffId, nid, maxNumofSplittable);

		if(pTempRChildStat[bufferPos].sum_hess == 0.0)//equivalent to IsEmpty()
		{
			pTempRChildStat[bufferPos].sum_gd += pGD[insId];
			pTempRChildStat[bufferPos].sum_hess += pHess[insId];
			pLastValue[bufferPos] = fvalue;
		}
		else
		{
			// try to find a split
			if(fabs(fvalue - pLastValue[bufferPos]) > rt_2eps)
			{
				float_point tempGD = pSNodeState[bufferPos].sum_gd - pTempRChildStat[bufferPos].sum_gd;
				float_point tempHess = pSNodeState[bufferPos].sum_hess - pTempRChildStat[bufferPos].sum_hess;
				bool needUpdate = NeedUpdate(pTempRChildStat[bufferPos].sum_hess, tempHess);
				if(needUpdate == true)
				{
					double sv = (fvalue + pLastValue[bufferPos]) * 0.5f;

		            UpdateSplitInfo(pSNodeState[bufferPos], pBestSplitPoint[bufferPos], pRChildStat[bufferPos],
		            							  pLChildStat[bufferPos], pTempRChildStat[bufferPos], tempGD, tempHess,
		            							  lambda, sv, featureId);
				}
			}
			//update the statistics
			pTempRChildStat[bufferPos].sum_gd += pGD[insId];
			pTempRChildStat[bufferPos].sum_hess += pHess[insId];
			pLastValue[bufferPos] = fvalue;
		}
	}

    // finish updating all statistics, check if it is possible to include all sum statistics
    for(int i = 0; i < numofSNode; i++)
    {
    	int buffId = pBuffId[i];//an id in the buffer
    	float_point tempGD = pSNodeState[buffId].sum_gd - pTempRChildStat[buffId].sum_gd;
    	float_point tempHess = pSNodeState[buffId].sum_hess - pTempRChildStat[buffId].sum_hess;
    	bool needUpdate = NeedUpdate(pTempRChildStat[buffId].sum_hess, tempHess);
        if(needUpdate == true)
        {
            const float delta = fabs(pLastValue[buffId]) + DeviceSplitter::rt_eps;
            float_point sv = pLastValue[buffId] + delta;

            UpdateSplitInfo(pSNodeState[buffId], pBestSplitPoint[buffId], pRChildStat[buffId], pLChildStat[buffId],
            							  pTempRChildStat[buffId], tempGD, tempHess, lambda, sv, featureId);
        }
    }
}

__device__ double CalGain(const nodeStat &parent, const nodeStat &r_child, float_point &l_child_GD,
										  float_point &l_child_Hess, float_point &lambda)
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

__device__ int GetBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable)
{
	int buffId = -1;

	int remain = snid % m_maxNumofSplittable;//use mode operation as Hash function to find the buffer position

	//checking where snid is located
	if(pSNIdToBuffId[remain] == snid)
	{
		buffId = remain;
	}
	else
	{
		//Hash conflict
		for(int i = m_maxNumofSplittable - 1; i > 0; i--)
		{
			if(pSNIdToBuffId[i] == snid)
				buffId = i;
		}
	}

	return buffId;
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

__device__ void UpdateLRStat(nodeStat &RChildStat, nodeStat &LChildStat, nodeStat &TempRChildStat,
											 float_point &grad, float_point &hess)
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

__device__ void UpdateSplitInfo(nodeStat &snStat, SplitPoint &bestSP, nodeStat &RChildStat, nodeStat &LChildStat,
											  nodeStat &TempRChildStat, float_point &tempGD, float_point &tempHess,
											  float_point &lambda, float_point &sv, int &featureId)
{
	double loss_chg = CalGain(snStat, TempRChildStat, tempGD, tempHess, lambda);
    bool bUpdated = UpdateSplitPoint(bestSP, loss_chg, sv, featureId);
	if(bUpdated == true)
	{
		UpdateLRStat(RChildStat, LChildStat, TempRChildStat, tempGD, tempHess);
	}
}

