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

__global__ void FindFeaSplitValue2(int *pnNumofKeyValues, long long *pnFeaStartPos, const int *pInsId, const float_point *pFeaValue, const int *pInsIdToNodeId,
								  nodeStat *pTempRChildStatPerThread, const float_point *pGD, const float_point *pHess, float_point *pLastValuePerThread,
								  nodeStat *pSNodeStatPerThread, SplitPoint *pBestSplitPointPerThread,
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

__global__ void PickBestFea(nodeStat *pTempRChildStatPerThread, float_point *pLastValuePerThread, nodeStat *pSNodeStatePerThread,
							SplitPoint *pBestSplitPointPerThread, nodeStat *pRChildStatPerThread, nodeStat *pLChildStatPerThread,
							int numofSNode, int numofFea, int maxNumofSplittable)
{
	//the best splittable node is stored in the first numofSNode nodes.
	for(int f = 1; f < numofFea; f++)
	{
		for(int n = 0; n < numofSNode; n++)
		{
			int nodePos = f * maxNumofSplittable + n;
			if(pBestSplitPointPerThread[nodePos].m_fGain > pBestSplitPointPerThread[n].m_fGain)
			{
				pBestSplitPointPerThread[n].m_fGain =pBestSplitPointPerThread[nodePos].m_fGain;
				pBestSplitPointPerThread[n].m_fSplitValue =pBestSplitPointPerThread[nodePos].m_fSplitValue;
				pBestSplitPointPerThread[n].m_nFeatureId =pBestSplitPointPerThread[nodePos].m_nFeatureId;
				pTempRChildStatPerThread[n].sum_gd = pTempRChildStatPerThread[nodePos].sum_gd;
				pTempRChildStatPerThread[n].sum_hess = pTempRChildStatPerThread[nodePos].sum_hess;
				pLastValuePerThread[n] = pLastValuePerThread[nodePos];
				pSNodeStatePerThread[n].sum_gd = pSNodeStatePerThread[nodePos].sum_gd;
				pSNodeStatePerThread[n].sum_hess = pSNodeStatePerThread[nodePos].sum_hess;
				pRChildStatPerThread[n].sum_gd = pRChildStatPerThread[nodePos].sum_gd;
				pRChildStatPerThread[n].sum_hess = pRChildStatPerThread[nodePos].sum_hess;
				pLChildStatPerThread[n].sum_gd = pLChildStatPerThread[nodePos].sum_gd;
				pLChildStatPerThread[n].sum_hess = pLChildStatPerThread[nodePos].sum_hess;
			}
		}
	}
}

__global__ void FindFeaSplitValue(int nNumofKeyValues, const int *idStartAddress, const float_point *pValueStartAddress, const int *pInsIdToNodeId,
								  nodeStat *pTempRChildStat, const float_point *pGD, const float_point *pHess, float_point *pLastValue,
								  nodeStat *pSNodeStat, SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat,
								  const int *pSNIdToBuffId, int maxNumofSplittable, int featureId, const int *pBuffId, int numofSNode, float_point lambda)
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
				float_point tempGD = pSNodeStat[bufferPos].sum_gd - pTempRChildStat[bufferPos].sum_gd;
				float_point tempHess = pSNodeStat[bufferPos].sum_hess - pTempRChildStat[bufferPos].sum_hess;
				bool needUpdate = NeedUpdate(pTempRChildStat[bufferPos].sum_hess, tempHess);
				if(needUpdate == true)
				{
					double sv = (fvalue + pLastValue[bufferPos]) * 0.5f;

		            UpdateSplitInfo(pSNodeStat[bufferPos], pBestSplitPoint[bufferPos], pRChildStat[bufferPos],
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
    	float_point tempGD = pSNodeStat[buffId].sum_gd - pTempRChildStat[buffId].sum_gd;
    	float_point tempHess = pSNodeStat[buffId].sum_hess - pTempRChildStat[buffId].sum_hess;
    	bool needUpdate = NeedUpdate(pTempRChildStat[buffId].sum_hess, tempHess);
        if(needUpdate == true)
        {
            const float delta = fabs(pLastValue[buffId]) + DeviceSplitter::rt_eps;
            float_point sv = pLastValue[buffId] + delta;

            UpdateSplitInfo(pSNodeStat[buffId], pBestSplitPoint[buffId], pRChildStat[buffId], pLChildStat[buffId],
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

