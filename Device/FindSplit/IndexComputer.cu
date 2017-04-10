/*
 * IndexComputer.cpp
 *
 *  Created on: 21 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: compute index for each feature value in the feature lists
 */

#include <vector>
#include <algorithm>
#include <cuda.h>
#include <helper_cuda.h>
#include "IndexComputer.h"
#include "../../DeviceHost/MyAssert.h"
#include "../Bagging/BagManager.h"
#include "../Hashing.h"
#include "../KernelConf.h"
#include "../../GetCudaError.h"
#include "../Memory/gbdtGPUMemManager.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

using std::vector;

int *IndexComputer::m_pInsId = NULL;	//instance id for each feature value in the feature lists
int IndexComputer::m_totalFeaValue = -1;//total number of feature values in the whole dataset
long long *IndexComputer::m_pFeaStartPos = NULL;//each feature start position
int IndexComputer::m_numFea = -1;	//number of features
int IndexComputer::m_maxNumofSN = -1;
long long IndexComputer::m_total_copy = -1;

long long *IndexComputer::m_pNumFeaValueEachNode_dh = NULL;	//# of feature values of each node

unsigned int *IndexComputer::m_pFvToInsId = NULL;

/**
  *@brief: mark feature values beloning to node with id=snId by 1
  */
__global__ void ArrayMarker(int *pBuffVec_d, unsigned int *pFvToInsId, int *pInsIdToNodeId, int totalNumFv,
							int maxNumSN, unsigned int *pSparseGatherIdx){
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= totalNumFv)//thread has nothing to mark 
		return;

	unsigned int insId = pFvToInsId[gTid];
	int nid = pInsIdToNodeId[insId];
	if(nid < 0)
		return;
	int buffId = nid % maxNumSN;
	unsigned int arrayId = blockIdx.z; //each arrayId corresponds to a prefix sum later
	int snId = pBuffVec_d[arrayId];
	int mark = (snId == buffId) ? 1 : 0;
	pSparseGatherIdx[gTid + arrayId * totalNumFv] = mark;
}

/**
 * @brief: compute feature value start position
 */
__global__ void ComputeFvalueStartPosEachNode(unsigned int *pnSparseGatherIdx, unsigned int totalFeaValue,
											  unsigned int numNode, long long *pFeaValueStartPosEachNode){
	unsigned int arrayStartPos = 0;//start position of this array (i.e. node)
	int i = 0;
	do{
		pFeaValueStartPosEachNode[i] = arrayStartPos;//feature value start position of each node
		i++;
		if(i >= numNode)
			break;
		arrayStartPos += pnSparseGatherIdx[i * totalFeaValue - 1];
	}while(true);
}

/**
  * @brief: store gather indices
  */
__global__ void CollectGatherIdx(unsigned int *pSparseGatherIdx, unsigned int totalNumFv,
								 long long *pEachFeaStartPosEachNode, unsigned int *pGatherIdx){
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= totalNumFv)//thread has nothing to mark
		return;

	unsigned int arrayId = blockIdx.z; //each arrayId corresponds to a prefix sum
	unsigned int arrayStartPos = pEachFeaStartPosEachNode[arrayId];//start position of this array (i.e. node)

	unsigned int idx = pSparseGatherIdx[gTid + arrayId * totalNumFv];
	if(gTid == 0){
		if(idx == 1)//store the first element
			pGatherIdx[gTid] = arrayStartPos + idx - 1;//set destination for element at gTid
		if(idx > 1)
			printf("error in CollectGatherIdx\n");
	}
	else{
		if(idx == pSparseGatherIdx[gTid + arrayId * totalNumFv - 1])//repeated element due to prefix sum
			return;
		pGatherIdx[gTid] = arrayStartPos + idx - 1;//set destimation for element at gTid
	}
}


/**
  *@brief: compute length and start position of each feature in each node
  */
__global__ void ComputeEachFeaInfo(long long *pEachFeaStartPos, int numFea, int totalFvalue, unsigned int *pSparseGatherIdx,
								   int *pEachFeaLenEachNode, long long *pEachFeaStartPosEachNode,
								   long long *pFeaValueStartPosEachNode, long long *pNumFeaValueEachNode){
	unsigned int arrayId = blockIdx.x; //each arrayId corresponds to a prefix sum
	int snId = arrayId;

	for(int f = 0; f < numFea; f++){
		unsigned int posOfLastFvalue;
		if(f < numFea - 1){
			PROCESS_ERROR(pEachFeaStartPos[f + 1] > 0);
			posOfLastFvalue = snId * totalFvalue + pEachFeaStartPos[f + 1] - 1;
		}
		else
			posOfLastFvalue = snId * totalFvalue + totalFvalue - 1;

		unsigned int startPos = pEachFeaStartPos[f];//start position of the feature f.
		unsigned int lenPreviousFvalue = 0;
		if(f > 0){
			lenPreviousFvalue = pSparseGatherIdx[snId * totalFvalue + startPos - 1];
		}
		pEachFeaLenEachNode[snId * numFea + f] = pSparseGatherIdx[posOfLastFvalue] - lenPreviousFvalue;
	}

	//start pos for first feature
	pEachFeaStartPosEachNode[snId * numFea] = pFeaValueStartPosEachNode[snId];
	//start pos for other feature
	for(int f = 1; f < numFea; f++){
		unsigned int feaPos = snId * numFea + f;
		pEachFeaStartPosEachNode[feaPos] = pEachFeaStartPosEachNode[feaPos - 1] + pEachFeaLenEachNode[feaPos];
	}

	//number of feature values of this node
	pNumFeaValueEachNode[snId] = pSparseGatherIdx[snId * totalFvalue + totalFvalue - 1];
}

/**
  * @brief: compute gether index by GPUs
  */
void IndexComputer::ComputeIdxGPU(int numSNode, int maxNumSN, int bagId){
	PROCESS_ERROR(m_pInsId != NULL && m_totalFeaValue > 0 && numSNode > 0);
	PROCESS_ERROR(maxNumSN >= 0 && maxNumSN == m_maxNumofSN);
	
	unsigned int *pnSparseGatherIdx;
	checkCudaErrors(cudaMalloc((void**)&pnSparseGatherIdx, sizeof(unsigned int) * m_totalFeaValue * numSNode));
	unsigned int *pnKey;
	checkCudaErrors(cudaMalloc((void**)&pnKey, sizeof(unsigned int) * m_totalFeaValue * numSNode));
	for(int i = 0; i < numSNode; i++){
		int flag = (i % 2 == 0 ? 0:(-1));
		checkCudaErrors(cudaMemset(pnKey + i * m_totalFeaValue, flag, sizeof(unsigned int) * m_totalFeaValue));
	}

	int flags = -1;//all bits are 1

	//memset for debuging; this should be removed to develop more reliable program
	memset(m_pNumFeaValueEachNode_dh, 0, sizeof(long long) * maxNumSN);

	KernelConf conf;
	int blockSizeForFvalue;
	dim3 dimNumofBlockForFvalue;
	conf.ConfKernel(m_totalFeaValue, blockSizeForFvalue, dimNumofBlockForFvalue);
	if(dimNumofBlockForFvalue.z > 1){
		printf("invalid kernel configuration!\n");
		exit(0);
	}
	dimNumofBlockForFvalue.z = numSNode;//each z value for a prefix sum.

	checkCudaErrors(cudaMemcpy(m_pFvToInsId, m_pInsId, sizeof(unsigned int) * m_totalFeaValue, cudaMemcpyHostToDevice));
	BagManager bagManager;
	int *pBuffVec_d = bagManager.m_pBuffIdVecEachBag + bagId * bagManager.m_maxNumSplittable;

	long long *pTmpFvalueStartPosEachNode = bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;

	int *pTmpInsIdToNodeId = bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns;
	ArrayMarker<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(pBuffVec_d, m_pFvToInsId, pTmpInsIdToNodeId,
																m_totalFeaValue, maxNumSN, pnSparseGatherIdx);
	GETERROR("after ArrayMarker");

	//compute prefix sum for one array
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey, pnKey + m_totalFeaValue * numSNode,
								  pnSparseGatherIdx, pnSparseGatherIdx);//in place prefix sum

	//get feature values start position of each node
	ComputeFvalueStartPosEachNode<<<1,1>>>(pnSparseGatherIdx, m_totalFeaValue, numSNode, pTmpFvalueStartPosEachNode);

	//write to gether index
	unsigned int *pTmpGatherIdx = bagManager.m_pIndicesEachBag_d + bagId * bagManager.m_numFeaValue;
	checkCudaErrors(cudaMemset(pTmpGatherIdx, flags, sizeof(unsigned int) * m_totalFeaValue));//when leaves appear, this is effective.
	CollectGatherIdx<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(pnSparseGatherIdx, m_totalFeaValue,
																	 pTmpFvalueStartPosEachNode, pTmpGatherIdx);
	GETERROR("after CollectGatherIdx");

	//compute each feature length and start position in each node
	GBDTGPUMemManager manager;
	int *pTmpEachFeaLenEachNode = bagManager.m_pEachFeaLenEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	long long * pTmpEachFeaStartPosEachNode = bagManager.m_pEachFeaStartPosEachNodeEachBag_d +
											  bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	ComputeEachFeaInfo<<<numSNode, 1>>>(manager.m_pFeaStartPos, m_numFea, m_totalFeaValue, pnSparseGatherIdx,
										pTmpEachFeaLenEachNode, pTmpEachFeaStartPosEachNode,
										pTmpFvalueStartPosEachNode, m_pNumFeaValueEachNode_dh);
	GETERROR("after ComputeEachFeaInfo");

	checkCudaErrors(cudaFree(pnSparseGatherIdx));
	checkCudaErrors(cudaFree(pnKey));
}

/**
 * @brief: allocate reusable memory
 */
void IndexComputer::AllocMem(int nNumofExamples, int nNumofFeatures, int maxNumofSplittableNode)
{
	m_numFea = nNumofFeatures;
	m_maxNumofSN = maxNumofSplittableNode;

	checkCudaErrors(cudaMallocHost((void**)&m_pNumFeaValueEachNode_dh, sizeof(long long) * m_maxNumofSN));

	checkCudaErrors(cudaMalloc((void**)&m_pFvToInsId, sizeof(unsigned int) * m_totalFeaValue));
}
