/*
 * DeviceSplitterKernel.cu
 *
 *  Created on: 10 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>

#include "FindFeaKernel.h"
#include "../Splitter/DeviceSplitter.h"
#include "../DeviceHashing.h"
#include "../prefix-sum/prefixSum.h"

const float rt_2eps = 2.0 * DeviceSplitter::rt_eps;

//helper functions on device
__device__ float_point CalGain(const nodeStat &parent, const nodeStat &r_child,
						  const float_point &l_child_GD, const float_point &l_child_Hess, const float_point &lambda);

__device__ bool UpdateSplitPoint(SplitPoint &curBest, float_point fGain, float_point fSplitValue, int nFeatureId);

__device__ void UpdateLRStat(nodeStat &RChildStat, nodeStat &LChildStat,
							 const nodeStat &TempRChildStat, const float_point &grad, const float_point &hess);
__device__ bool NeedUpdate(float_point &RChildHess, float_point &LChildHess);
__device__ void UpdateSplitInfo(const nodeStat &snStat, SplitPoint &bestSP, nodeStat &RChildStat, nodeStat &LChildStat,
								const nodeStat &TempRChildStat, const float_point &tempGD, const float_point &temHess,
								const float_point &lambda, const float_point &sv, const int &featureId);

/**
 * @brief: each thread processes one feature
 */
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
		float_point fvalue = pInsValueStartAddress[i];

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
					float_point sv = (fvalue + pLastValuePerThread[bufferPos]) * 0.5f;
					if(hashValue == 1)
					{
//						float_point loss_chg = CalGain(pSNodeStatPerThread[bufferPos], pTempRChildStatPerThread[bufferPos], tempGD, tempHess, lambda);
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
	int feaId = blockIdx.y + smallestFeaId;//###### need to add a shift here to process only part of the features
	int numofValuePerFea = gridDim.x * blockDim.x;
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
	if(tidForEachFeaValue >= numofValuePerFea)
	{
		printf("numofValuePerFea is smaller than the numof threads!\n");
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
		int bufferPos = snId * numofValuePerFea * feaBatch + feaId * numofValuePerFea + tidForEachFeaValue;
		if(pGDOnEachFeaValue[bufferPos] != 0 || pHessOnEachFeaValue[bufferPos]!= 0 || pValueOneEachFeaValue[bufferPos] != 0)
			printf("default value of gd/hess/fvalue is incorrect in ObtainGDEachNode.\n");

		//GD/Hess of the same node is stored consecutively.
		pGDOnEachFeaValue[bufferPos] = pGD[insId];
		pHessOnEachFeaValue[bufferPos] = pHess[insId];
		pValueOneEachFeaValue[bufferPos] = pInsValueStartAddress[tidForEachFeaValue];
	}
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

__global__ void ComputeGain(const nodeStat *pSNodeStatPerThread, int feaBatch, float_point *pGDOnEachFeaValue_d,
							float_point *pHessOnEachFeaValue_d, const int *pnStartPosEachFeaInBatch, const int *pnEachFeaLen)
{
	int nGlobalThreadId = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
}


__device__ float_point CalGain(const nodeStat &parent, const nodeStat &r_child,
						  const float_point &l_child_GD, const float_point &l_child_Hess,
						  const float_point &lambda)
{
//	PROCESS_ERROR(abs(parent.sum_gd - l_child_GD - r_child.sum_gd) < 0.0001);
//	PROCESS_ERROR(parent.sum_hess == l_child_Hess + r_child.sum_hess);

//	printf("lgd=%f, lhe=%f, rgd=%f, rhe=%f, pgd=%f, phe=%f, lamb=%f\n", l_child_GD, l_child_Hess,
//			r_child.sum_gd, r_child.sum_hess, parent.sum_gd, parent.sum_hess, lambda);

	//compute the gain
	float_point fGain = (l_child_GD * l_child_GD)/(l_child_Hess + lambda) +
				   (r_child.sum_gd * r_child.sum_gd)/(r_child.sum_hess + lambda) -
				   (parent.sum_gd * parent.sum_gd)/(parent.sum_hess + lambda);
//	if(fGain > -10)
//	{
//		printf("gain=%f, lgd=%f, lhe=%f, rgd=%f, rhe=%f, pgd=%f, phe=%f, lamb=%f\n", fGain, l_child_GD, l_child_Hess,
//				r_child.sum_gd, r_child.sum_hess, parent.sum_gd, parent.sum_hess, lambda);
//	}


	return fGain;
}


 __device__ bool UpdateSplitPoint(SplitPoint &curBest, float_point fGain, float_point fSplitValue, int nFeatureId)
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
	float_point loss_chg = CalGain(snStat, TempRChildStat, tempGD, tempHess, lambda);
    bool bUpdated = UpdateSplitPoint(bestSP, loss_chg, sv, featureId);
	if(bUpdated == true)
	{
		UpdateLRStat(RChildStat, LChildStat, TempRChildStat, tempGD, tempHess);
	}
}

