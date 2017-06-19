/*
 * DeviceSplitterKernel.h
 *
 *  Created on: 10 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITTERKERNEL_H_
#define DEVICESPLITTERKERNEL_H_

#include "../Splitter/DeviceSplitter.h"
#include "../../Host/UpdateOps/SplitPoint.h"
#include "../../DeviceHost/NodeStat.h"
#include "../../DeviceHost/BaseClasses/BaseSplitter.h"
#include "../../SharedUtility/DataType.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/binarySearch.h"

//dense array
__global__ void LoadGDHessFvalueRoot(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue);
__global__ void LoadGDHessFvalue(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, const uint *pDstIndexEachFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue);
__global__ void ComputeGainDense(const nodeStat *pSNodeStat, const int *pBuffId,	real lambda,
							const double *pGDPrefixSumOnEachFeaValue, const real *pHessPrefixSumOnEachFeaValue,
							const real *pDenseFeaValue, int numofDenseValue,
							const uint *pEachFeaStartEachNode, const int *pEachFeaLenEachNode,
							const uint *pnKey, int numFea, real *pGainOnEachFeaValue, bool *pDefault2Right);
__global__ void FirstFeaGain(const unsigned int *pEachFeaStartPosEachNode, int numFeaStartPos, real *pGainOnEachFeaValue, uint numFeaValue);
__global__ void PickLocalBestSplitEachNode(const uint *pnNumFeaValueEachNode, const uint *pFeaStartPosEachNode,
										   const real *pGainOnEachFeaValue,
								   	   	   real *pfLocalBestGain, int *pnLocalBestGainKey);
__global__ void PickGlobalBestSplitEachNode(const real *pfLocalBestGain, const int *pnLocalBestGainKey,
								   	   	    real *pfGlobalBestGain, int *pnGlobalBestGainKey,
								   	   	    int numBlockPerNode, int numofSNode);
/**
 * @brief: find split points
 */
template<class T>
__global__ void FindSplitInfo(const uint *pEachFeaStartPosEachNode, const T *pEachFeaLenEachNode,
							  const real *pDenseFeaValue, const real *pfGlobalBestGain, const T *pnGlobalBestGainKey,
							  const int *pPartitionId2SNPos, const int numFea,
							  const nodeStat *snNodeStat, const double *pPrefixSumGD, const real *pPrefixSumHess,
							  const bool *pDefault2Right, const uint *pnKey,
							  SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat)
{
	//a thread for constructing a split point
	int snId = threadIdx.x;//position in the dense array of nodes
	T key = pnGlobalBestGainKey[snId];//position in the dense array

	//find best feature id
	uint bestFeaId = -1;
	RangeBinarySearch(key, pEachFeaStartPosEachNode + snId * numFea, numFea, bestFeaId);

	CONCHECKER(bestFeaId != -1);

	int snPos = pPartitionId2SNPos[snId];//snId to buffer id (i.e. hash value)

	pBestSplitPoint[snPos].m_fGain = pfGlobalBestGain[snId];//change the gain back to positive
	if(pfGlobalBestGain[snId] <= 0){//no gain
		return;
	}

	pBestSplitPoint[snPos].m_nFeatureId = bestFeaId;
	ECHECKER(key);
	pBestSplitPoint[snPos].m_fSplitValue = 0.5f * (pDenseFeaValue[key] + pDenseFeaValue[key - 1]);
	pBestSplitPoint[snPos].m_bDefault2Right = false;

	//child node stat
	int idxPreSum = key - 1;//follow xgboost using exclusive
	if(pDefault2Right[key] == false){
		pLChildStat[snPos].sum_gd = snNodeStat[snPos].sum_gd - pPrefixSumGD[idxPreSum];
		pLChildStat[snPos].sum_hess = snNodeStat[snPos].sum_hess - pPrefixSumHess[idxPreSum];
		pRChildStat[snPos].sum_gd = pPrefixSumGD[idxPreSum];
		pRChildStat[snPos].sum_hess = pPrefixSumHess[idxPreSum];
	}
	else{
		pBestSplitPoint[snPos].m_bDefault2Right = true;

		real parentGD = snNodeStat[snPos].sum_gd;
		real parentHess = snNodeStat[snPos].sum_hess;

		uint segId = pnKey[key];
		uint segStartPos = pEachFeaStartPosEachNode[segId];
		T segLen = pEachFeaLenEachNode[segId];
		uint lastFvaluePos = segStartPos + segLen - 1;
		real totalMissingGD = parentGD - pPrefixSumGD[lastFvaluePos];
		real totalMissingHess = parentHess - pPrefixSumHess[lastFvaluePos];

		double rChildGD = totalMissingGD + pPrefixSumGD[idxPreSum];
		real rChildHess = totalMissingHess + pPrefixSumHess[idxPreSum];
		real lChildGD = parentGD - rChildGD;
		real lChildHess = parentHess - rChildHess;

		pRChildStat[snPos].sum_gd = rChildGD;
		pRChildStat[snPos].sum_hess = rChildHess;
		pLChildStat[snPos].sum_gd = lChildGD;
		pLChildStat[snPos].sum_hess = lChildHess;
	}
	ECHECKER(pLChildStat[snPos].sum_hess);
	ECHECKER(pRChildStat[snPos].sum_hess);
	printf("split: f=%d, value=%f, gain=%f, gd=%f v.s. %f, hess=%f v.s. %f, buffId=%d, key=%d\n", bestFeaId, pBestSplitPoint[snPos].m_fSplitValue,
			pBestSplitPoint[snPos].m_fGain, pLChildStat[snPos].sum_gd, pRChildStat[snPos].sum_gd, pLChildStat[snPos].sum_hess, pRChildStat[snPos].sum_hess, snPos, key);
}

//helper functions
template<class T>
__device__ bool NeedUpdate(T &RChildHess, T &LChildHess)
{
	if(LChildHess >= DeviceSplitter::min_child_weight && RChildHess >= DeviceSplitter::min_child_weight)
		return true;
	return false;
}


#endif /* DEVICESPLITTERKERNEL_H_ */
