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
#include "../../SharedUtility/getMin.h"
#include "../../SharedUtility/DataType.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/binarySearch.h"
#include "../../SharedUtility/DeviceUtility.h"

//dense array
__global__ void LoadGDHessFvalueRoot(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, int numFeaValue, double *pGDEachFeaValue, real *pHessEachFeaValue);
__global__ void LoadGDHessFvalue(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, const uint *pDstIndexEachFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue, uint*);

__global__ void FirstFeaGain(const unsigned int *pEachFeaStartPosEachNode, int numFeaStartPos, real *pGainOnEachFeaValue, uint numFeaValue);

//helper functions
__device__ bool NeedCompGain(double RChildHess, double LChildHess);
__device__ double ComputeGain(double tempGD, double tempHess, real lambda, double rChildGD, double rChildHess, double parentGD, double parentHess);

template<class T>
__global__ void ComputeGainDense(const nodeStat *pSNodeStat, const int *pid2SNPos, real lambda,
							const double *pGDPrefixSumOnEachFeaValue, const real *pHessPrefixSumOnEachFeaValue,
							const real *pDenseFeaValue, int numofDenseValue,
							const uint *pEachFeaStartEachNode, const T *pEachFeaLenEachNode,
							const uint *pnKey, int numFea, real *pGainOnEachFeaValue, bool *pDefault2Right)
{
	const float rt_2eps = 2.0 * DeviceSplitter::rt_eps;
	//one thread loads one value
	uint gTid = GLOBAL_TID();
	if(gTid >= numofDenseValue)//the thread has no gain to compute, i.e. a thread per gain
		return;

	uint segId = pnKey[gTid];
	uint pid = segId / numFea;

	int snPos = pid2SNPos[pid];
	ECHECKER(snPos);

    int segLen = pEachFeaLenEachNode[segId];
    uint segStartPos = pEachFeaStartEachNode[segId];
	uint lastFvaluePos = segStartPos + segLen - 1;
	double parentGD = pSNodeStat[snPos].sum_gd;
	double parentHess = pSNodeStat[snPos].sum_hess;
	double totalMissingGD = parentGD - pGDPrefixSumOnEachFeaValue[lastFvaluePos];
	double totalMissingHess = parentHess - pHessPrefixSumOnEachFeaValue[lastFvaluePos];

	if(gTid == segStartPos){//include all sum statistics; store the gain at the first pos of each segment
    	//check default to left
		double totalGD = pGDPrefixSumOnEachFeaValue[lastFvaluePos];
		double totalHess = pHessPrefixSumOnEachFeaValue[lastFvaluePos];
		bool needUpdate = NeedCompGain(totalHess, totalMissingHess);
    	real all2Left = 0, all2Right = -1.0;
    	if(needUpdate == true){
    		CONCHECKER(totalHess > 0);
    		CONCHECKER(totalMissingHess > 0);
    		all2Left = ComputeGain(totalMissingGD, totalMissingHess, lambda, totalGD, totalHess, parentGD, parentHess);
    		all2Right = ComputeGain(totalGD, totalHess, lambda, totalMissingGD, totalMissingHess, parentGD, parentHess);
    	}

    	//check default to right
    	if(all2Left < all2Right){
    		pGainOnEachFeaValue[gTid] = all2Right;
    		pDefault2Right[gTid] = true;
    	}
    	else
    		pGainOnEachFeaValue[gTid] = all2Left;
	}
	else{
		//if the previous fea value is the same as the current fea value, gain is 0 for the current fea value.
		real preFvalue = pDenseFeaValue[gTid - 1], curFvalue = pDenseFeaValue[gTid];
		if(preFvalue - curFvalue <= rt_2eps && preFvalue - curFvalue >= -rt_2eps)//############## backwards is not considered!
		{//avoid same feature value different gain issue
			pGainOnEachFeaValue[gTid] = 0;
			return;
		}

		int exclusiveSumPos = gTid - 1;//following xgboost using exclusive sum on gd and hess

		//forward consideration (fvalues are sorted descendingly)
		double rChildGD = pGDPrefixSumOnEachFeaValue[exclusiveSumPos];
		double rChildHess = pHessPrefixSumOnEachFeaValue[exclusiveSumPos];
		double tempGD = parentGD - rChildGD;
		double tempHess = parentHess - rChildHess;
		bool needUpdate = NeedCompGain(rChildHess, tempHess);
		if(needUpdate == true)//need to compute the gain
		{
			ECHECKER(tempHess > 0);
			ECHECKER(parentHess > 0);
			double tempGain = ComputeGain(tempGD, tempHess, lambda, rChildGD, rChildHess, parentGD, parentHess);
			pGainOnEachFeaValue[gTid] = tempGain;
		}
		else{
			//assign gain to 0
			pGainOnEachFeaValue[gTid] = 0;
		}

		//backward consideration
		if(totalMissingHess < 1)//there is no instance with missing values
			return;
		//missing values to the right child
		rChildGD += totalMissingGD;
		rChildHess += totalMissingHess;
		tempGD = parentGD - rChildGD;
		tempHess = parentHess - rChildHess;
		needUpdate = NeedCompGain(rChildHess, tempHess);
		if(needUpdate == true){
			ECHECKER(tempHess > 0);
			ECHECKER(parentHess > 0);
			double tempGain = ComputeGain(tempGD, tempHess, lambda, rChildGD, rChildHess, parentGD, parentHess);

			if(tempGain > 0 && tempGain - pGainOnEachFeaValue[gTid] > 0.1){
				pGainOnEachFeaValue[gTid] = tempGain;
				pDefault2Right[gTid] = true;
			}
		}
	}//end of forward and backward consideration
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
	int pid = threadIdx.x;//position in the dense array of nodes
	int snPos = pPartitionId2SNPos[pid];//snId to buffer id (i.e. hash value)
	ECHECKER(snPos);
	pBestSplitPoint[snPos].m_fGain = pfGlobalBestGain[pid];//change the gain back to positive
	if(pfGlobalBestGain[pid] <= DeviceSplitter::rt_eps)//no gain
		return;

	T key = pnGlobalBestGainKey[pid];//position in the dense array
	//find best feature id
	uint segId = pnKey[key];
	uint bestFeaId = segId % numFea;
	CONCHECKER(bestFeaId < numFea);
	pBestSplitPoint[snPos].m_nFeatureId = bestFeaId;
	pBestSplitPoint[snPos].m_bDefault2Right = false;

	//handle all to left/right case
	uint segStartPos = pEachFeaStartPosEachNode[segId];
	T segLen = pEachFeaLenEachNode[segId];
	uint lastFvaluePos = segStartPos + segLen - 1;
	if(key == 0 || (key > 0 && pnKey[key] != pnKey[key - 1])){//first element of the feature
        const real gap = fabs(pDenseFeaValue[key]) + DeviceSplitter::rt_eps;
		if(pDefault2Right[key] == true){//all non-missing to left
			pBestSplitPoint[snPos].m_bDefault2Right = true;
			pBestSplitPoint[snPos].m_fSplitValue = pDenseFeaValue[lastFvaluePos] + gap;
			pLChildStat[snPos].sum_gd = pPrefixSumGD[lastFvaluePos];
			pLChildStat[snPos].sum_hess = pPrefixSumHess[lastFvaluePos];
			pRChildStat[snPos].sum_gd = snNodeStat[snPos].sum_gd - pPrefixSumGD[lastFvaluePos];
			pRChildStat[snPos].sum_hess = snNodeStat[snPos].sum_hess - pPrefixSumHess[lastFvaluePos];
		}
		else{//all non-missing to right
			pBestSplitPoint[snPos].m_fSplitValue = pDenseFeaValue[lastFvaluePos] - gap;
			pLChildStat[snPos].sum_gd = snNodeStat[snPos].sum_gd - pPrefixSumGD[lastFvaluePos];
			pLChildStat[snPos].sum_hess = snNodeStat[snPos].sum_hess - pPrefixSumHess[lastFvaluePos];
			pRChildStat[snPos].sum_gd = pPrefixSumGD[lastFvaluePos];
			pRChildStat[snPos].sum_hess = pPrefixSumHess[lastFvaluePos];
		}
	}
	else{//non-first element of the feature
		pBestSplitPoint[snPos].m_fSplitValue = 0.5f * (pDenseFeaValue[key] + pDenseFeaValue[key - 1]);

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

			real totalMissingGD = parentGD - pPrefixSumGD[lastFvaluePos];
			real totalMissingHess = parentHess - pPrefixSumHess[lastFvaluePos];

			double rChildGD = totalMissingGD + pPrefixSumGD[idxPreSum];
			real rChildHess = totalMissingHess + pPrefixSumHess[idxPreSum];
			ECHECKER(rChildHess);
			real lChildGD = parentGD - rChildGD;
			real lChildHess = parentHess - rChildHess;
			ECHECKER(lChildHess);

			pRChildStat[snPos].sum_gd = rChildGD;
			pRChildStat[snPos].sum_hess = rChildHess;
			pLChildStat[snPos].sum_gd = lChildGD;
			pLChildStat[snPos].sum_hess = lChildHess;
		}
		ECHECKER(pLChildStat[snPos].sum_hess);
		ECHECKER(pRChildStat[snPos].sum_hess);
	}

//	printf("split: f=%d, value=%f, gain=%f, gd=%f v.s. %f, hess=%f v.s. %f, buffId=%d, key=%d, pid=%d, df2Left=%d\n", bestFeaId, pBestSplitPoint[snPos].m_fSplitValue,
//			pBestSplitPoint[snPos].m_fGain, pLChildStat[snPos].sum_gd, pRChildStat[snPos].sum_gd, pLChildStat[snPos].sum_hess,
//			pRChildStat[snPos].sum_hess, snPos, key, pid, pDefault2Right[key]);
}

#endif /* DEVICESPLITTERKERNEL_H_ */
