/*
 * ComputeGainDense.cu
 *
 *  Created on: 22 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: kernels gain computing using dense arrays
 */

#include <stdio.h>
#include <float.h>
#include <limits>
#include "FindFeaKernel.h"
#include "../KernelConst.h"
#include "../../DeviceHost/svm-shared/DeviceUtility.h"
#include "../Splitter/DeviceSplitter.h"

const float rt_2eps = 2.0 * DeviceSplitter::rt_eps;

#define testing


__global__ void ComputeIndex(int *pDstIndexEachFeaValue, long long totalFeaValue)
{
	long long gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= totalFeaValue)
		return;
	pDstIndexEachFeaValue[gTid] = gTid;
}

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void LoadGDHessFvalueRoot(const float_point *pInsGD, const float_point *pInsHess, int numIns,
						   const int *pInsId, const float_point *pAllFeaValue, int numFeaValue,
						   float_point *pGDEachFeaValue, float_point *pHessEachFeaValue, float_point *pDenseFeaValue)
{
	//one thread loads one value
	//## global id looks ok, but need to be careful
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	int insId = pInsId[gTid];//instance id

#ifdef testing
	if(insId >= numIns)
		printf("Instance id is larger than the number of instances!\n");
#endif

	//store GD and Hess.
	pGDEachFeaValue[gTid] = pInsGD[insId];
	pHessEachFeaValue[gTid] = pInsHess[insId];
	pDenseFeaValue[gTid] = pAllFeaValue[gTid];
}

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void LoadGDHessFvalue(const float_point *pInsGD, const float_point *pInsHess, int numIns,
						   const int *pInsId, const float_point *pAllFeaValue, const int *pDstIndexEachFeaValue, int numFeaValue,
						   float_point *pGDEachFeaValue, float_point *pHessEachFeaValue, float_point *pDenseFeaValue)
{
	//one thread loads one value
	//## global id looks ok, but need to be careful
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	int insId = pInsId[gTid];//instance id

#ifdef testing
	if(insId >= numIns)
		printf("Instance id is larger than the number of instances!\n");
#endif

	//index for scatter
	int idx = pDstIndexEachFeaValue[gTid];
	if(idx == -1)//instance is in a leaf node
		return;

#ifdef testing
	if(idx < 0)
		printf("index to out array is negative!\n");
	if(idx >= numFeaValue)
		printf("index to out array is too large: %d. numFvalue=%d!\n", idx, numFeaValue);
#endif

	//scatter: store GD, Hess and the feature value.
	pGDEachFeaValue[idx] = pInsGD[insId];
	pHessEachFeaValue[idx] = pInsHess[insId];
	pDenseFeaValue[idx] = pAllFeaValue[gTid];
}

/**
 * @brief: compute the gain in parallel, each gain is computed by a thread.
 */
__global__ void ComputeGainDense(const nodeStat *pSNodeStat, const long long *pFeaValueStartPosEachNode, int numSN,
							const int *pBuffId, float_point lambda,
							const float_point *pGDPrefixSumOnEachFeaValue, const float_point *pHessPrefixSumOnEachFeaValue,
							const float_point *pDenseFeaValue, int numofDenseValue, float_point *pGainOnEachFeaValue)
{
	//one thread loads one value
	long long gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	//compute node id
	int snId = -1;
	for(int i = 0; i < numSN; i++)
	{
		if(i == numSN - 1)
		{
			snId = i;
			break;
		}
		else if(gTid >= pFeaValueStartPosEachNode[i] && gTid < pFeaValueStartPosEachNode[i + 1])
		{
			snId = i;
			break;
		}
	}
	int hashVaue = pBuffId[snId];
	if(hashVaue < 0)
		printf("Error in ComputeGain: buffer id %d, i=%d\n", hashVaue, snId);

	if(gTid >= numofDenseValue)//the thread has no gain to compute, i.e. a thread per gain
	{
		return;
	}

	if(gTid == 0)
	{
		//assign gain to 0 to the first feature value
    	pGainOnEachFeaValue[gTid] = 0;
		return;
	}

	//if the previous fea value is the same as the current fea value, gain is 0 for the current fea value.
	if(fabs(pDenseFeaValue[gTid - 1] - pDenseFeaValue[gTid]) <= rt_2eps)
	{//avoid same feature value different gain issue
		pGainOnEachFeaValue[gTid] = 0;
		return;
	}

	int exclusiveSumPos = gTid - 1;//following xgboost using exclusive sum on gd and hess

	float_point rChildGD = pGDPrefixSumOnEachFeaValue[exclusiveSumPos];
	float_point rChildHess = pHessPrefixSumOnEachFeaValue[exclusiveSumPos];
	float_point parentGD = pSNodeStat[hashVaue].sum_gd;
	float_point parentHess = pSNodeStat[hashVaue].sum_hess;
	float_point tempGD = parentGD - rChildGD;
	float_point tempHess = parentHess - rChildHess;
	bool needUpdate = NeedUpdate(rChildHess, tempHess);
    if(needUpdate == true)//need to compute the gain
    {
		float_point tempGain = (tempGD * tempGD)/(tempHess + lambda) + 
						  	   (rChildGD * rChildGD)/(rChildHess + lambda) -
	  						   (parentGD * parentGD)/(parentHess + lambda);
    	pGainOnEachFeaValue[gTid] = tempGain; 
//    	if(pGainOnEachFeaValue[gTid] > 0 && ((rChildHess == 463714 && tempHess == 1) || (rChildHess == 1 && tempHess == 463714)))
 //   		printf("gain=%f, gid=%d, rhess=%f, lhess=%f\n", pGainOnEachFeaValue[gTid], gTid, rChildHess, tempHess);
    }
    else
    {
    	//assign gain to 0
    	pGainOnEachFeaValue[gTid] = 0;
    }
}

/**
 * @brief: change the gain of the first value of each feature to 0
 */
__global__ void FirstFeaGain(const long long *pEachFeaStartPosEachNode, int numFeaStartPos, float_point *pGainOnEachFeaValue)
{
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	if(gTid >= numFeaStartPos)//no gain to fix
	{
		return;
	}
	long long gainPos = pEachFeaStartPosEachNode[gTid];
//	printf("gTid=%d, gainPos=%ld\n", gTid, gainPos);
//	printf("change %f to 0 pos at %d, gainPos=%ld\n", pGainOnEachFeaValue[gainPos], pEachFeaStartPosEachNode[gTid], gainPos);
	pGainOnEachFeaValue[gainPos] = 0;
//	if(gTid == 0){
//		printf("pEachFeaStartPosEachNode[8]=%f\n", pEachFeaStartPosEachNode[8]);
//	}
}

/**
 * @brief: pick best feature of this batch for all the splittable nodes
 * Each block.y processes one node, a thread processes a reduction.
 */
__global__ void PickLocalBestSplitEachNode(const long long *pnNumFeaValueEachNode, const long long *pFeaStartPosEachNode,
										   const float_point *pGainOnEachFeaValue,
								   	   	   float_point *pfLocalBestGain, int *pnLocalBestGainKey)
{
	//best gain of each node is search by a few blocks
	//blockIdx.z corresponds to a splittable node id
	int snId = blockIdx.z;

	__shared__ float_point pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
	int localTid = threadIdx.x;
	pfGain[localTid] = FLT_MAX;//initialise to a large positive number
	pnBetterGainKey[localTid] = -1;
	if(localTid == 0){//initialise local best value
		int blockId = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
		pfLocalBestGain[blockId] = FLT_MAX;
		pnLocalBestGainKey[blockId] = -1;
	}

	long long numValueThisNode = pnNumFeaValueEachNode[snId];//get the number of feature value of this node
	long long tidForEachNode = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	long long nPos = pFeaStartPosEachNode[snId] + tidForEachNode;//feature value gain position
	if(nPos < 0)
		printf("sp pos is nagative! %d\n", nPos);
	if(tidForEachNode >= numValueThisNode){//no gain to load
		return;
	}
	pfGain[localTid] = -pGainOnEachFeaValue[nPos];//change to find min of -gain
	pnBetterGainKey[localTid] = nPos;//############ need to be long long
	__syncthreads();

	//find the local best split point
	GetMinValue(pfGain, pnBetterGainKey, blockDim.x);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		int blockId = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
		pfLocalBestGain[blockId] = pfGain[0];
		pnLocalBestGainKey[blockId] = pnBetterGainKey[0];
		if(pnBetterGainKey[0] < 0)
			printf("negative key: snId=%d, blockId=%d, gain=%f, key=%d\n", snId, blockId, pfGain[0], pnBetterGainKey[0]);
	}
}

/**
 * @brief: pick best feature of this batch for all the splittable nodes
 */
__global__ void PickGlobalBestSplitEachNode(const float_point *pfLocalBestGain, const int *pnLocalBestGainKey,
								   	   	    float_point *pfGlobalBestGain, int *pnGlobalBestGainKey,
								   	   	    int numBlockPerNode, int numofSNode)
{
	//a block for finding the best gain of a node
	int blockId = blockIdx.x;

	int snId = blockId;
	if(blockIdx.y > 1)
		printf("One block is not enough to find global best split.\n");

	if(snId >= numofSNode)
		printf("Error in PickBestFea: kernel split %d nods, but only %d splittable nodes\n", snId, numofSNode);

	__shared__ float_point pfGain[BLOCK_SIZE];
	__shared__ int pnBetterGainKey[BLOCK_SIZE];
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
	GetMinValue(pfGain, pnBetterGainKey, blockDim.x);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		pfGlobalBestGain[snId] = -pfGain[0];//make the gain back to its original sign
		pnGlobalBestGainKey[snId] = pnBetterGainKey[0];
		if(pnBetterGainKey[0] < 0)
			printf("negative key: snId=%d, gain=%f, key=%d, blockDim.x=%d, blockSize=%d, blockpPerNode=%d\n", snId, pfGain[0], pnBetterGainKey[0], blockDim.x, BLOCK_SIZE, numBlockPerNode);
	}
}

/**
 * @brief: find split points
 */
__global__ void FindSplitInfo(const long long *pEachFeaStartPosEachNode, const int *pEachFeaLenEachNode,
							  const float_point *pDenseFeaValue, const float_point *pfGlobalBestGain, const int *pnGlobalBestGainKey,
							  const int *pPosToBuffId, const int numFea,
							  const nodeStat *snNodeStat, const float_point *pPrefixSumGD, const float_point *pPrefixSumHess,
							  SplitPoint *pBestSplitPoint, nodeStat *pRChildStat, nodeStat *pLChildStat)
{
	//a thread for constructing a split point
	int snId = threadIdx.x;//position in the dense array of nodes
	int key = pnGlobalBestGainKey[snId];//position in the dense array

	//find best feature id
	int bestFeaId = -1;
	for(int f = 0; f < numFea; f++)
	{
		int feaPos = f + snId * numFea;
		int numofFValue = pEachFeaLenEachNode[feaPos];
		if(pEachFeaStartPosEachNode[feaPos] + numofFValue < key)//####### key should be represented using long long
			continue;
		else//key is in the range of values of f
		{
			bestFeaId = f;
			break;
		}
	}

	if(bestFeaId == -1)
		printf("Error: bestFeaId=%d\n", bestFeaId);

	int buffId = pPosToBuffId[snId];//snId to buffer id (i.e. hash value)

	pBestSplitPoint[buffId].m_fGain = pfGlobalBestGain[snId];//change the gain back to positive
	if(pfGlobalBestGain[snId] <= 0){//no gain
		return;
	}

	pBestSplitPoint[buffId].m_nFeatureId = bestFeaId;
	if(key < 1)
		printf("Error: best key=%d, is < 1\n", key);
	pBestSplitPoint[buffId].m_fSplitValue = 0.5f * (pDenseFeaValue[key] + pDenseFeaValue[key - 1]);

	//child node stat
	int idxPreSum = key - 1;//follow xgboost using exclusive
	pLChildStat[buffId].sum_gd = snNodeStat[buffId].sum_gd - pPrefixSumGD[idxPreSum];
	pLChildStat[buffId].sum_hess = snNodeStat[buffId].sum_hess - pPrefixSumHess[idxPreSum];
//	if(pLChildStat[buffId].sum_hess == 1)
//		printf("Have a look at here\n");
	pRChildStat[buffId].sum_gd = pPrefixSumGD[idxPreSum];
	pRChildStat[buffId].sum_hess = pPrefixSumHess[idxPreSum];
	if(pLChildStat[buffId].sum_hess < 0 || pRChildStat[buffId].sum_hess < 0)
		printf("Error: hess is negative l hess=%d, r hess=%d\n", pLChildStat[buffId].sum_hess, pRChildStat[buffId].sum_hess);
//	printf("split: f=%d, value=%f, gain=%f, gd=%f v.s. %f, hess=%f v.s. %f, buffId=%d, key=%d\n", bestFeaId, pBestSplitPoint[buffId].m_fSplitValue,
//			pBestSplitPoint[buffId].m_fGain, pLChildStat[buffId].sum_gd, pRChildStat[buffId].sum_gd, pLChildStat[buffId].sum_hess, pRChildStat[buffId].sum_hess, buffId, key);
}
