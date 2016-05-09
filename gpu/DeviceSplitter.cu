/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "DeviceSplitter.h"
#include <algorithm>
#include <math.h>
#include <map>
#include <iostream>

#include "../pureHost/MyAssert.h"
#include "gbdtGPUMemManager.h"


using std::map;
using std::make_pair;
using std::cout;
using std::endl;

const float rt_2eps = 2.0 * DeviceSplitter::rt_eps;
__global__ void FindFeaSplitValue(int nNumofKeyValues, int *idStartAddress, float_point *pValueStartAddress, int *pInsIdToNodeId,
								  nodeStat *pTempRChildStat, float_point *pGD, float_point *pHess, float_point *pLastValue,
								  nodeStat *pSNodeState, SplitPoint *pBestSplitPoin, nodeStat *pRChildStat, nodeStat *pLChildStat,
								  int *pSNIdToBuffId, int maxNumofSplittable, int featureId, int numofSNode, float_point lambda);

/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{

	int numofSNode = vBest.size();

	GBDTGPUMemManager manager;


	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	for(int f = 0; f < nNumofFeature; f++)
	{

		int *pInsId = manager.pDInsId;
		float_point *pFeaValue = manager.pdDFeaValue;
		int *pNumofKeyValue = manager.pDNumofKeyValue;

		int numofCurFeaKeyValues = -1;
		//the number of key values of the f{th} feature
		manager.MemcpyDeviceToHost(&numofCurFeaKeyValues, pNumofKeyValue + f, sizeof(int));
		PROCESS_ERROR(numofCurFeaKeyValues > 0);

		long long startPosOfPrevFea = 0;
		if(f > 0)
		{
			//copy value of the start position of the previous feature
			manager.MemcpyDeviceToHost(&startPosOfPrevFea, manager.pFeaStartPos + (f - 1), sizeof(long long));
		}
		PROCESS_ERROR(startPosOfPrevFea >= 0);
		long long startPosOfCurFea = startPosOfPrevFea + numofCurFeaKeyValues;
		//copy the value of the start position of the current feature
		manager.MemcpyHostToDevice(manager.pFeaStartPos + f, &startPosOfCurFea, sizeof(long long));

		int *pInsToNodeId = new int[manager.m_numofIns];
		PROCESS_ERROR(manager.m_numofIns == m_nodeIds.size());
		manager.VecToArray(m_nodeIds, pInsToNodeId);
		manager.MemcpyHostToDevice(manager.pInsIdToNodeId, pInsToNodeId, sizeof(int) * manager.m_numofIns);
		nodeStat *pHostNodeStat = new nodeStat[manager.m_numofIns];
		manager.VecToArray(m_nodeStat, pHostNodeStat);
		manager.MemcpyHostToDevice(manager.pSNodeStat, pHostNodeStat, sizeof(int) * manager.m_numofIns);

		float_point *pHostGD = new float_point[manager.m_numofIns];
		float_point *pHostHess = new float_point[manager.m_numofIns];
		for(int i = 0; i < manager.m_numofIns; i++)
		{
			pHostGD[i] = m_vGDPair_fixedPos[i].grad;
			pHostHess[i] = m_vGDPair_fixedPos[i].hess;
		}
		manager.MemcpyHostToDevice(manager.pGrad, pHostGD, manager.m_numofIns);
		manager.MemcpyHostToDevice(manager.pHess, pHostHess, manager.m_numofIns);

		delete[] pHostGD;
		delete[] pHostHess;
		delete[] pInsToNodeId;
		delete[] pHostNodeStat;

nodeStat *pSNodeState = manager.pSNodeStat;
nodeStat *pTempRChildStat = manager.pTempRChildStat;
float_point *pLastValue = manager.pLastValue;

float_point *pGD = manager.pGrad;
float_point *pHess = manager.pHess;

		//find the split value for this feature
	int *idStartAddress = pInsId + startPosOfCurFea;
	float_point *pValueStartAddress = pFeaValue + startPosOfCurFea;

		FindFeaSplitValue<<<1, 1>>>(numofCurFeaKeyValues, idStartAddress, pValueStartAddress, manager.pInsIdToNodeId,
									pTempRChildStat, pGD, pHess, pLastValue, pSNodeState, manager.pBestSplitPoint,
									manager.pRChildStat, manager.pLChildStat, manager.pSNIdToBuffId,
									manager.m_maxNumofSplittable, f, numofSNode, DeviceSplitter::m_labda);
	}
}

__global__ void FindFeaSplitValue(int nNumofKeyValues, int *idStartAddress, float_point *pValueStartAddress, int *pInsIdToNodeId,
								  nodeStat *pTempRChildStat, float_point *pGD, float_point *pHess, float_point *pLastValue,
								  nodeStat *pSNodeState, SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat,
								  int *pSNIdToBuffId, int maxNumofSplittable, int featureId, int numofSNode, float_point lambda)
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
		int bufferPos = DeviceSplitter::GetBufferId(pSNIdToBuffId, nid, maxNumofSplittable);

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
				bool needUpdate = DeviceSplitter::NeedUpdate(pTempRChildStat[bufferPos].sum_hess, tempHess);
				if(needUpdate == true)
				{
					double sv = (fvalue + pLastValue[bufferPos]) * 0.5f;

		            DeviceSplitter::UpdateSplitInfo(pSNodeState[bufferPos], pBestSplitPoint[bufferPos], pRChildStat[bufferPos],
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
    	float_point tempGD = pSNodeState[i].sum_gd - pTempRChildStat[i].sum_gd;
    	float_point tempHess = pSNodeState[i].sum_hess - pTempRChildStat[i].sum_hess;
    	bool needUpdate = DeviceSplitter::NeedUpdate(pTempRChildStat[i].sum_hess, tempHess);
        if(needUpdate == true)
        {
            const float delta = fabs(pLastValue[i]) + DeviceSplitter::rt_eps;
            float_point sv = pLastValue[i] + delta;

            DeviceSplitter::UpdateSplitInfo(pSNodeState[i], pBestSplitPoint[i], pRChildStat[i], pLChildStat[i],
            							  pTempRChildStat[i], tempGD, tempHess, lambda, sv, featureId);
        }
    }
}

__device__ double DeviceSplitter::CalGain(const nodeStat &parent, const nodeStat &r_child, float_point &l_child_GD,
										  float_point &l_child_Hess, float_point &lambda)
{
	PROCESS_ERROR(abs(parent.sum_gd - l_child_GD - r_child.sum_gd) < 0.0001);
	PROCESS_ERROR(parent.sum_hess == l_child_Hess + r_child.sum_hess);

	//compute the gain
	double fGain = (l_child_GD * l_child_GD)/(l_child_Hess + lambda) +
				   (r_child.sum_gd * r_child.sum_gd)/(r_child.sum_hess + lambda) -
				   (parent.sum_gd * parent.sum_gd)/(parent.sum_hess + lambda);

	return fGain;
}

__device__ int DeviceSplitter::GetBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable)
{
	int buffId = -1;

	int remain = snid % m_maxNumofSplittable;//use mode operation as Hash function to find the buffer position
	if(pSNIdToBuffId[remain] == -1)
		buffId = remain;
	else
	{
		//Hash conflict
		for(int i = m_maxNumofSplittable - 1; i > 0; i--)
		{
			if(pSNIdToBuffId[i] == -1)
				buffId = i;
		}
	}

	return buffId;
}

 __device__ bool DeviceSplitter::UpdateSplitPoint(SplitPoint &curBest, double fGain, double fSplitValue, int nFeatureId)
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

__device__ void DeviceSplitter::UpdateLRStat(nodeStat &RChildStat, nodeStat &LChildStat, nodeStat &TempRChildStat,
											 float_point &grad, float_point &hess)
{
	LChildStat.sum_gd = grad;
	LChildStat.sum_hess = hess;
	RChildStat = TempRChildStat;
}

__device__ bool DeviceSplitter::NeedUpdate(float_point &RChildHess, float_point &LChildHess)
{
	if(LChildHess >= DeviceSplitter::min_child_weight && RChildHess >= DeviceSplitter::min_child_weight)
		return true;
	return false;
}

__device__ void DeviceSplitter::UpdateSplitInfo(nodeStat &snStat, SplitPoint &bestSP, nodeStat &RChildStat, nodeStat &LChildStat,
											  nodeStat &TempRChildStat, float_point &tempGD, float_point &tempHess,
											  float_point &lambda, float_point &sv, int &featureId)
{
	double loss_chg = CalGain(snStat, TempRChildStat, tempGD, tempHess, lambda);
    bool bUpdated = DeviceSplitter::UpdateSplitPoint(bestSP, loss_chg, sv, featureId);
	if(bUpdated == true)
	{
		DeviceSplitter::UpdateLRStat(RChildStat, LChildStat, TempRChildStat, tempGD, tempHess);
	}
}
