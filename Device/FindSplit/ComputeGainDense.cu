/*
 * ComputeGainDense.cu
 *
 *  Created on: 22 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: kernels gain computing using dense arrays
 */

#include <stdio.h>
#include <float.h>
#include "FindFeaKernel.h"
#include "../KernelConst.h"
#include "../svm-shared/DeviceUtility.h"

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void LoadGDHess(const float_point *pInsGD, const float_point *pInsHess, int numIns,
						   const int *pInsId, const int *pDstIndexEachFeaValue, int numFeaValue,
						   float_point *pGDEachFeaValue, float_point *pHessEachFeaValue)
{
	//one thread loads one value
	//## global id looks ok, but need to be careful
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	int insId = pInsId[gTid];//instance id
	if(insId >= numIns)
		printf("Instance id is larger than the number of instances!\n");

	int idx = pDstIndexEachFeaValue[gTid];
	if(idx == -1)//instance is in a leaf node
		return;

	if(idx < 0)
		printf("index to out array is negative!\n");
	if(idx >= numFeaValue)
		printf("index to out array is too large!\n");

	//store GD and Hess.
	pGDEachFeaValue[idx] = pInsGD[insId];
	pHessEachFeaValue[idx] = pInsHess[insId];
}

/**
 * @brief: compute the gain in parallel, each gain is computed by a thread.
 */
__global__ void ComputeGainDense(const nodeStat *pSNodeStat, const int *pFeaValueStartPosEachNode, int numSN,
							const int *pBuffId,	float_point lambda,
							const float_point *pGDPrefixSumOnEachFeaValue, const float_point *pHessPrefixSumOnEachFeaValue,
							int numofDenseValue, float_point *pGainOnEachFeaValue)
{
	//one thread loads one value
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

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

	if(gTid >= numofDenseValue)//the thread has no gain to compute
	{
		return;
	}

	if(gTid == 0)
	{
		//assign gain to 0
    	pGainOnEachFeaValue[gTid] = 0;
		return;
	}

	int exclusiveSumPos = gTid - 1;//following xgboost using exclusive sum on gd and hess

	float_point rChildGD = pGDPrefixSumOnEachFeaValue[exclusiveSumPos];
	float_point rChildHess = pHessPrefixSumOnEachFeaValue[exclusiveSumPos];
	float_point snGD = pSNodeStat[hashVaue].sum_gd;
	float_point snHess = pSNodeStat[hashVaue].sum_hess;
	float_point tempGD = snGD - rChildGD;
	float_point tempHess = snHess - rChildHess;
	bool needUpdate = NeedUpdate(rChildHess, tempHess);
    if(needUpdate == true)//need to compute the gain
    {
    	pGainOnEachFeaValue[gTid] = (tempGD * tempGD)/(tempHess + lambda) +
    									 	 (rChildGD * rChildGD)/(rChildHess + lambda) -
    									 	 (snGD * snGD)/(snHess + lambda);
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
__global__ void FirstFeaGain(const int *pEachFeaStartPosEachNode, int numFeaStartPos, float_point *pGainOnEachFeaValue)
{
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	if(gTid >= numFeaStartPos)//no gain to fix
	{
		return;
	}
	int gainPos = pEachFeaStartPosEachNode[gTid];
	pGainOnEachFeaValue[gainPos] = 0;
}

/**
 * @brief: pick best feature of this batch for all the splittable nodes
 * Each block.y processes one node, a thread processes a reduction.
 */
__global__ void PickLocalBestSplitEachNode(const int *pnNumFeaValueEachNode, const int *pFeaStartPosEachNode,
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
	if(localTid == 0)
	{//initialise local best value
		int blockId = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
		pfLocalBestGain[blockId] = FLT_MAX;
		pnLocalBestGainKey[blockId] = -1;
	}

	int numValueThisNode = pnNumFeaValueEachNode[snId];//get the number of feature value of this node
	int tidForEachNode = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	if(tidForEachNode >= numValueThisNode)//no gain to load
	{
		return;
	}

	int nPos = pFeaStartPosEachNode[snId] + tidForEachNode;//feature value gain position
	if(nPos < 0)
		printf("sp pos is nagative! %d\n", nPos);

	pfGain[localTid] = -pGainOnEachFeaValue[nPos];//change to find min of -gain
	pnBetterGainKey[localTid] = nPos;//############ need to be the key in the whole fea value array
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
			printf("negative key: snId=%d, gain=%f, key=%d\n", snId, pfGain[0], pnBetterGainKey[0]);
	}
}
