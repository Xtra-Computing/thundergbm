/*
 * DeviceSplitterKernel.h
 *
 *  Created on: 10 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITTERKERNEL_H_
#define DEVICESPLITTERKERNEL_H_

#include <float.h>
#include <limits>

#include "../Splitter/DeviceSplitter.h"
#include "../../Host/UpdateOps/SplitPoint.h"
#include "../../DeviceHost/NodeStat.h"
#include "../../DeviceHost/BaseClasses/BaseSplitter.h"
#include "../../DeviceHost/svm-shared/DeviceUtility.h"
#include "../../SharedUtility/getMin.h"
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

template<class T>
__global__ void PickLocalBestSplitEachNode(const uint *pnNumFeaValueEachNode, const uint *pFeaStartPosEachNode,
										   const real *pGainOnEachFeaValue,
								   	   	   real *pfLocalBestGain, T *pnLocalBestGainKey)
{
	//best gain of each node is search by a few blocks
	//blockIdx.z corresponds to a splittable node id
	int snId = blockIdx.z;
	uint numValueThisNode = pnNumFeaValueEachNode[snId];//get the number of feature value of this node
	int blockId = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
	uint tid0 = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
	if(tid0 >= numValueThisNode){
		pfLocalBestGain[blockId] = 0;
		pnLocalBestGainKey[blockId] = tid0;
		return;
	}

	__shared__ real pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[localTid] = -1;
	if(localTid == 0){//initialise local best value
		pfLocalBestGain[blockId] = FLT_MAX;
		pnLocalBestGainKey[blockId] = -1;
	}

	uint tidForEachNode = tid0 + threadIdx.x;
	uint nPos = pFeaStartPosEachNode[snId] + tidForEachNode;//feature value gain position


	if(tidForEachNode >= numValueThisNode){//no gain to load
		pfGain[localTid] = 0;
		pnBetterGainKey[localTid] = INT_MAX;
	}
	else{
		pfGain[localTid] = -pGainOnEachFeaValue[nPos];//change to find min of -gain
		pnBetterGainKey[localTid] = nPos;
	}
	__syncthreads();

	//find the local best split point
	GetMinValueOriginal(pfGain, pnBetterGainKey);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		pfLocalBestGain[blockId] = pfGain[0];
		pnLocalBestGainKey[blockId] = pnBetterGainKey[0];

		ECHECKER(pnBetterGainKey[0]);
		//if(pnBetterGainKey[0] < 0)
		//	printf("negative key: snId=%d, blockId=%d, gain=%f, key=%d\n", snId, blockId, pfGain[0], pnBetterGainKey[0]);
	}
}
template<class T>
__global__ void PickGlobalBestSplitEachNode(const real *pfLocalBestGain, const T *pnLocalBestGainKey,
								   	   	    real *pfGlobalBestGain, T *pnGlobalBestGainKey,
								   	   	    int numBlockPerNode, int numofSNode)
{
	//a block for finding the best gain of a node
	int blockId = blockIdx.x;

	int snId = blockId;
	CONCHECKER(blockIdx.y <= 1);
	CONCHECKER(snId < numofSNode);

	__shared__ real pfGain[BLOCK_SIZE];
	__shared__ T pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[localTid] = -1;

	if(localTid >= numBlockPerNode)//number of threads is larger than the number of blocks
	{
		return;
	}

	int curFeaLocalBestStartPos = snId * numBlockPerNode;

	LoadToSharedMem(numBlockPerNode, curFeaLocalBestStartPos, pfLocalBestGain, pnLocalBestGainKey, pfGain, pnBetterGainKey);
	 __syncthreads();	//wait until the thread within the block

	//find the local best split point
	GetMinValueOriginal(pfGain, pnBetterGainKey);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		pfGlobalBestGain[snId] = -pfGain[0];//make the gain back to its original sign
		pnGlobalBestGainKey[snId] = pnBetterGainKey[0];
	}
}
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
//	printf("split: f=%d, value=%f, gain=%f, gd=%f v.s. %f, hess=%f v.s. %f, buffId=%d, key=%d\n", bestFeaId, pBestSplitPoint[snPos].m_fSplitValue,
//			pBestSplitPoint[snPos].m_fGain, pLChildStat[snPos].sum_gd, pRChildStat[snPos].sum_gd, pLChildStat[snPos].sum_hess, pRChildStat[snPos].sum_hess, snPos, key);
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
