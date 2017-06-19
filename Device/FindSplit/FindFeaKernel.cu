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
#include "../Splitter/DeviceSplitter.h"
#include "../../DeviceHost/svm-shared/DeviceUtility.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/getMin.h"
#include "../../SharedUtility/binarySearch.h"

const float rt_2eps = 2.0 * DeviceSplitter::rt_eps;

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void LoadGDHessFvalueRoot(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue)
{
	//one thread loads one value
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	int insId = pInsId[gTid];//instance id

	CONCHECKER(insId < numIns);

	//store GD and Hess.
	pGDEachFeaValue[gTid] = pInsGD[insId];
	pHessEachFeaValue[gTid] = pInsHess[insId];
	pDenseFeaValue[gTid] = pAllFeaValue[gTid];
}

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void LoadGDHessFvalue(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, const unsigned int *pDstIndexEachFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue)
{
	//one thread loads one value
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	int insId = pInsId[gTid];//instance id

	CONCHECKER(insId < numIns);

	//index for scatter
	int idx = pDstIndexEachFeaValue[gTid];
	if(idx == -1)//instance is in a leaf node
		return;

	CONCHECKER(idx >= 0);
	CONCHECKER(idx < numFeaValue);

	//scatter: store GD, Hess and the feature value.
	pGDEachFeaValue[idx] = pInsGD[insId];
	pHessEachFeaValue[idx] = pInsHess[insId];
	pDenseFeaValue[idx] = pAllFeaValue[gTid];
}

/**
 * @brief: compute the gain in parallel, each gain is computed by a thread.
 */
__global__ void ComputeGainDense(const nodeStat *pSNodeStat, const int *pId2SNPos, real lambda,
							const double *pGDPrefixSumOnEachFeaValue, const real *pHessPrefixSumOnEachFeaValue,
							const real *pDenseFeaValue, int numofDenseValue,
							const uint *pEachFeaStartEachNode, const int *pEachFeaLenEachNode,
							const uint *pnKey, int numFea, real *pGainOnEachFeaValue, bool *pDefault2Right)
{
	//one thread loads one value
	uint gTid = GLOBAL_TID();
	if(gTid >= numofDenseValue)//the thread has no gain to compute, i.e. a thread per gain
		return;

	uint segId = pnKey[gTid];
	uint pid = segId / numFea;

	int snPos = pId2SNPos[pid];
	ECHECKER(snPos);

	if(gTid == 0)
	{
		//assign gain to 0 to the first feature value
    	pGainOnEachFeaValue[gTid] = 0;
		return;
	}

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
	double parentGD = pSNodeStat[snPos].sum_gd;
	double parentHess = pSNodeStat[snPos].sum_hess;
	double tempGD = parentGD - rChildGD;
	double tempHess = parentHess - rChildHess;
	bool needUpdate = NeedUpdate(rChildHess, tempHess);
    if(needUpdate == true)//need to compute the gain
    {
		double tempGain = (tempGD * tempGD)/(tempHess + lambda) +
						  	   (rChildGD * rChildGD)/(rChildHess + lambda) -
	  						   (parentGD * parentGD)/(parentHess + lambda);
    	pGainOnEachFeaValue[gTid] = tempGain; 
    }
    else{
    	//assign gain to 0
    	pGainOnEachFeaValue[gTid] = 0;
    }

    //backward consideration
    int segLen = pEachFeaLenEachNode[segId];
    uint segStartPos = pEachFeaStartEachNode[segId];
    uint lastFvaluePos = segStartPos + segLen - 1;
    double totalMissingGD = parentGD - pGDPrefixSumOnEachFeaValue[lastFvaluePos];
    double totalMissingHess = parentHess - pHessPrefixSumOnEachFeaValue[lastFvaluePos];
    if(totalMissingHess < 1)//there is no instance with missing values
    	return;
    //missing values to the right child
    rChildGD += totalMissingGD;
    rChildHess += totalMissingHess;
    tempGD = parentGD - rChildGD;
    tempHess = parentHess - rChildHess;
    needUpdate = NeedUpdate(rChildHess, tempHess);
    if(needUpdate == true){
    	double tempGain = (tempGD * tempGD)/(tempHess + lambda) +
			  	   	    (rChildGD * rChildGD)/(rChildHess + lambda) -
			  	   	    (parentGD * parentGD)/(parentHess + lambda);

    	if(tempGain > 0 && tempGain - pGainOnEachFeaValue[gTid] > 0.1){
    		pGainOnEachFeaValue[gTid] = tempGain;
    		pDefault2Right[gTid] = true;
    	}
    }
}

/**
 * @brief: change the gain of the first value of each feature to 0
 */
__global__ void FirstFeaGain(const unsigned int *pEachFeaStartPosEachNode, int numFeaStartPos, real *pGainOnEachFeaValue, uint numFeaValue)
{
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaStartPos)//no gain to fix
		return;
	unsigned int gainPos = pEachFeaStartPosEachNode[gTid];
	if(gainPos >= numFeaValue)
		return;//there may be some ending 0s (e.g. the last node has some features with any values).
	pGainOnEachFeaValue[gainPos] = 0;
}

/**
 * @brief: pick best feature of this batch for all the splittable nodes
 * Each block.y processes one node, a thread processes a reduction.
 */
__global__ void PickLocalBestSplitEachNode(const uint *pnNumFeaValueEachNode, const uint *pFeaStartPosEachNode,
										   const real *pGainOnEachFeaValue,
								   	   	   real *pfLocalBestGain, int *pnLocalBestGainKey)
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

/**
 * @brief: pick best feature of this batch for all the splittable nodes
 */
__global__ void PickGlobalBestSplitEachNode(const real *pfLocalBestGain, const int *pnLocalBestGainKey,
								   	   	    real *pfGlobalBestGain, int *pnGlobalBestGainKey,
								   	   	    int numBlockPerNode, int numofSNode)
{
	//a block for finding the best gain of a node
	int blockId = blockIdx.x;

	int snId = blockId;
	CONCHECKER(blockIdx.y <= 1);
	CONCHECKER(snId < numofSNode);

	__shared__ real pfGain[BLOCK_SIZE];
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
	GetMinValueOriginal(pfGain, pnBetterGainKey);
	__syncthreads();
	if(localTid == 0)//copy the best gain to global memory
	{
		pfGlobalBestGain[snId] = -pfGain[0];//make the gain back to its original sign
		pnGlobalBestGainKey[snId] = pnBetterGainKey[0];
		ECHECKER(pnBetterGainKey[0]);
		if(pnBetterGainKey[0] < 0)
			printf("negative key: snId=%d, gain=%f, key=%d, blockDim.x=%d, blockSize=%d, blockpPerNode=%d, numSN=%d\n",
			snId, pfGain[0], pnBetterGainKey[0], blockDim.x, BLOCK_SIZE, numBlockPerNode, numofSNode);
	}
}
