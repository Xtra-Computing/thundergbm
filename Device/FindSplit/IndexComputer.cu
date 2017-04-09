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
#include "../prefix-sum/prefixSum.h"
#include "../../GetCudaError.h"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

using std::vector;

int *IndexComputer::m_pInsId = NULL;	//instance id for each feature value in the feature lists
int IndexComputer::m_totalFeaValue = -1;//total number of feature values in the whole dataset
long long *IndexComputer::m_pFeaStartPos = NULL;//each feature start position
int IndexComputer::m_numFea = -1;	//number of features
int IndexComputer::m_maxNumofSN = -1;
long long IndexComputer::m_total_copy = -1;

int *IndexComputer::m_insIdToNodeId_dh = NULL;//instance id to node id
int *IndexComputer::m_pIndices_dh = NULL;	//index for each node
long long *IndexComputer::m_pNumFeaValueEachNode_dh = NULL;	//# of feature values of each node
long long *IndexComputer::m_pFeaValueStartPosEachNode_dh = NULL;//start positions to feature value of each node
long long *IndexComputer::m_pEachFeaStartPosEachNode_dh = NULL;//each feature start position in each node
int *IndexComputer::m_pEachFeaLenEachNode_dh = NULL;//each feature value length in each node
int *IndexComputer::m_pBuffIdToPos_dh = NULL;//map buff id to dense pos id; not all elements in this array are used, due to not continuous buffid.

unsigned int *IndexComputer::m_pEachFeaStartPos_dh = NULL;
unsigned int *IndexComputer::m_pnGatherIdx = NULL;
unsigned int *IndexComputer::m_pFvToInsId = NULL;

/**
  *@brief: mark feature values beloning to node with id=snId by 1
  */
__global__ void ArrayMarker(int *pBuffVec_d, unsigned int *pFvToInsId, int *pInsIdToNodeId, int totalNumFv, int maxNumSN, unsigned int *pSparseGatherIdx){
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
	if(snId == buffId){
		pSparseGatherIdx[gTid + arrayId * totalNumFv] = 1;
	}
}

/**
  *@brief: compute length for each feature value of each node 
  */
void UpdateEachFeaLenEachNode(unsigned int *pEachFeaStartPos, int snId, int numFea, int totalFvalue, unsigned int *pSparseGatherIdx, int *pEachFeaLenEachNode){
	for(int f = 0; f < numFea; f++){
		unsigned int posOfLastFvalue;
		if(f < numFea - 1){
			PROCESS_ERROR(pEachFeaStartPos[f + 1] > 0);
			posOfLastFvalue = pEachFeaStartPos[f + 1] - 1;
		}
		else
			posOfLastFvalue = totalFvalue - 1;

		unsigned int startPos = pEachFeaStartPos[f];//start position of the feature f.
		unsigned int lenPreviousFvalue = 0;
		if(f > 0){
			lenPreviousFvalue = pSparseGatherIdx[startPos - 1];
		}
		pEachFeaLenEachNode[snId * numFea + f] = pSparseGatherIdx[posOfLastFvalue] - lenPreviousFvalue;
	}
}

/**
  *@brief: compute start position of each feature in each node
  */
void ComputeEachFeaStartPosEachNode(int numFea, int snId, unsigned int collectedGatherIdx, int *pEachFeaLenEachNode, long long *pEachFeaStartPosEachNode){
	//start pos for first feature
	pEachFeaStartPosEachNode[snId * numFea] = collectedGatherIdx;
	//start pos for other feature
	for(int f = 1; f < numFea; f++){
		unsigned int feaPos = snId * numFea + f;
		pEachFeaStartPosEachNode[feaPos] = pEachFeaStartPosEachNode[feaPos - 1] + pEachFeaLenEachNode[feaPos];
	}
}

/**
  *@brief: compute length and start position of each feature in each node
  */
__global__ void ComputeEachFeaInfo(unsigned int *pEachFeaStartPos, int numFea, int totalFvalue, unsigned int *pSparseGatherIdx, int *pEachFeaLenEachNode,
								   long long *pEachFeaStartPosEachNode, long long *pFeaValueStartPosEachNode, long long *pNumFeaValueEachNode){
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

	unsigned int arrayStartPos = 0;//start position of this array (i.e. node)
	for(int i = 1; i < arrayId; i++){//will improve it later
		arrayStartPos += pSparseGatherIdx[i * totalFvalue - 1];	
	}
	//start pos for first feature
	pEachFeaStartPosEachNode[snId * numFea] = arrayStartPos;
	//start pos for other feature
	for(int f = 1; f < numFea; f++){
		unsigned int feaPos = snId * numFea + f;
		pEachFeaStartPosEachNode[feaPos] = pEachFeaStartPosEachNode[feaPos - 1] + pEachFeaLenEachNode[feaPos];
	}

	//feature value start position of each node 
	pFeaValueStartPosEachNode[snId] = arrayStartPos;

	//number of feature values of this node
	pNumFeaValueEachNode[snId] = pSparseGatherIdx[snId * totalFvalue + totalFvalue - 1];
}

/**
  * @brief: store gather indices
  */
__global__ void CollectGatherIdx(unsigned int *pSparseGatherIdx, unsigned int totalNumFv, unsigned int *pGatherIdx){
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= totalNumFv)//thread has nothing to mark 
		return;

	unsigned int arrayId = blockIdx.z; //each arrayId corresponds to a prefix sum 
	unsigned int arrayStartPos = 0;//start position of this array (i.e. node)
	for(int i = 1; i < arrayId; i++){//will improve it later
		arrayStartPos += pSparseGatherIdx[i * totalNumFv - 1];	
	}
	unsigned int idx = pSparseGatherIdx[gTid + arrayId * totalNumFv];
	if(gTid == 0){
		if(idx == 1)//store the first element
			pGatherIdx[gTid] = arrayStartPos + idx - 1;//set destination for element at gTid
		if(idx > 1 || idx < 0)
			printf("error in CollectGatherIdx\n");
	}
	else{
		if(idx == pSparseGatherIdx[gTid - 1])//repeated element due to prefix sum
			return;
		pGatherIdx[gTid] = arrayStartPos + idx - 1;//set destimation for element at gTid
	}
}

__global__ void FloatToUnsignedInt(float_point *pfSparseGatherIdx, unsigned int totalNumFv, unsigned int *pnSparseGatherIdx){
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= totalNumFv)//thread has nothing to mark 
		return;
	pnSparseGatherIdx[gTid] = pfSparseGatherIdx[gTid];
}



/**
  * @brief: compute gether index by GPUs
  */
void IndexComputer::ComputeIdxGPU(int numSNode, int maxNumSN, const int *pBuffVec){
	PROCESS_ERROR(m_pInsId != NULL && m_totalFeaValue > 0 && m_insIdToNodeId_dh != NULL);
	PROCESS_ERROR(numSNode > 0 && m_pIndices_dh != NULL);
	PROCESS_ERROR(maxNumSN >= 0);
	PROCESS_ERROR(maxNumSN == m_maxNumofSN);
	
	unsigned int *pnKey;
	unsigned int *pnSparseGatherIdx;
	checkCudaErrors(cudaMalloc((void**)&pnSparseGatherIdx, sizeof(unsigned int) * m_totalFeaValue * numSNode));
	checkCudaErrors(cudaMalloc((void**)&pnKey, sizeof(unsigned int) * m_totalFeaValue * numSNode));
	for(int i = 0; i < numSNode; i++){
		int flag = (i % 2 == 0 ? 0:-1);
		checkCudaErrors(cudaMemset(pnKey + i * m_totalFeaValue, flag, sizeof(unsigned int) * m_totalFeaValue));
	}
	checkCudaErrors(cudaMemset(pnSparseGatherIdx, 0, sizeof(unsigned int) * m_totalFeaValue * numSNode));

	unsigned int flags = -1;//all bits are 1
	checkCudaErrors(cudaMemset(m_pnGatherIdx, flags, sizeof(unsigned int) * m_totalFeaValue));//when leaves appear, this is effective.

	//memset for debuging; this should be removed to develop more reliable program
	memset(m_pEachFeaLenEachNode_dh, 0, sizeof(int) * maxNumSN * m_numFea);
	memset(m_pNumFeaValueEachNode_dh, 0, sizeof(long long) * maxNumSN);
	memset(m_pEachFeaStartPosEachNode_dh, 0, sizeof(long long) * m_numFea * maxNumSN);
	memset(m_pFeaValueStartPosEachNode_dh, 0, sizeof(long long) * m_numFea * maxNumSN);
	GETERROR("after memset for idx comp");
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
	int *pBuffVec_d;
	checkCudaErrors(cudaMalloc((void**)pBuffVec_d, sizeof(int) * numSNode));
	checkCudaErrors(cudaMemcpy(pBuffVec_d, pBuffVec, sizeof(int) * numSNode, cudaMemcpyHostToDevice));

	ArrayMarker<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(pBuffVec_d, m_pFvToInsId, m_insIdToNodeId_dh, m_totalFeaValue, maxNumSN, pnSparseGatherIdx);
	GETERROR("after ArrayMarker");

	//compute prefix sum for one array
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey, pnKey + m_totalFeaValue * numSNode, pnSparseGatherIdx, pnSparseGatherIdx);//in place prefix sum
	//write to gether index
	CollectGatherIdx<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(pnSparseGatherIdx, m_totalFeaValue, m_pnGatherIdx);

	//compute each feature length and start position in each node
	ComputeEachFeaInfo<<<numSNode, 1>>>(m_pEachFeaStartPos_dh, m_numFea, m_totalFeaValue, pnSparseGatherIdx, m_pEachFeaLenEachNode_dh, m_pEachFeaStartPosEachNode_dh,
										m_pFeaValueStartPosEachNode_dh, m_pNumFeaValueEachNode_dh);

	checkCudaErrors(cudaFree(pnSparseGatherIdx));
}


__global__ void llToUint(long long *pLlArray, unsigned int *pUintArray, int numEle){
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= numEle)//thread has nothing to mark 
		return;
	pUintArray[gTid] = pLlArray[gTid];

}
/**
 * @brief: convert copy long long to unsigned int
 */
void IndexComputer::LonglongToUnsignedInt(long long *pFeaStartPos, unsigned int *pEachFeaStartPos_dh, int numEle){
	KernelConf conf;
	int blockSize;
	dim3 dimNumofBlock;
	conf.ConfKernel(numEle, blockSize, dimNumofBlock);

	llToUint<<<dimNumofBlock, blockSize>>>(pFeaStartPos, pEachFeaStartPos_dh, numEle);
}


/**
 * @brief: allocate reusable memory
 */
void IndexComputer::AllocMem(int nNumofExamples, int nNumofFeatures, int maxNumofSplittableNode)
{
	m_numFea = nNumofFeatures;
	m_maxNumofSN = maxNumofSplittableNode;

	checkCudaErrors(cudaMallocHost((void**)&m_pIndices_dh, sizeof(int) * m_totalFeaValue));
	checkCudaErrors(cudaMallocHost((void**)&m_insIdToNodeId_dh, sizeof(int) * nNumofExamples));
	checkCudaErrors(cudaMallocHost((void**)&m_pNumFeaValueEachNode_dh, sizeof(long long) * m_maxNumofSN));
	checkCudaErrors(cudaMallocHost((void**)&m_pBuffIdToPos_dh, sizeof(int) * m_maxNumofSN));
	checkCudaErrors(cudaMallocHost((void**)&m_pFeaValueStartPosEachNode_dh, sizeof(long long) * m_numFea * m_maxNumofSN));
	checkCudaErrors(cudaMallocHost((void**)&m_pEachFeaStartPosEachNode_dh, sizeof(long long) * m_maxNumofSN * m_numFea));
	checkCudaErrors(cudaMallocHost((void**)&m_pEachFeaLenEachNode_dh, sizeof(int) * m_maxNumofSN * m_numFea));

	checkCudaErrors(cudaMallocHost((void**)&m_pEachFeaStartPos_dh, sizeof(unsigned int) * m_numFea));
	checkCudaErrors(cudaMalloc((void**)&m_pnGatherIdx, sizeof(unsigned int) * m_totalFeaValue));
	checkCudaErrors(cudaMalloc((void**)&m_pFvToInsId, sizeof(unsigned int) * m_totalFeaValue));
}
