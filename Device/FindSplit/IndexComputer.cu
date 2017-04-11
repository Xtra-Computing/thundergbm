/*
 * IndexComputer.cpp
 *
 *  Created on: 21 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: compute index for each feature value in the feature lists
 */

#include <cuda.h>
#include <vector>
#include <algorithm>
#include <helper_cuda.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "IndexComputer.h"
#include "../Hashing.h"
#include "../KernelConf.h"
#include "../Bagging/BagManager.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../DeviceHost/MyAssert.h"
#include "../../SharedUtility/GetCudaError.h"
#include "../../SharedUtility/KernelMacro.h"
#include "../../SharedUtility/HostUtility.h"

using std::vector;

int IndexComputer::m_totalFeaValue = -1;//total number of feature values in the whole dataset
long long *IndexComputer::m_pFeaStartPos = NULL;//each feature start position
int IndexComputer::m_numFea = -1;	//number of features
int IndexComputer::m_maxNumofSN = -1;
long long IndexComputer::m_total_copy = -1;

long long *IndexComputer::m_pNumFeaValueEachNode_dh = NULL;	//# of feature values of each node
unsigned int *IndexComputer::pPartitionMarker = NULL;
unsigned int *IndexComputer::pnKey = NULL;

/**
  *@brief: mark feature values beloning to node with id=snId by 1
  */
__global__ void MarkPartition(unsigned int *pBuffId2PartitionId, int *pFvToInsId, int *pInsIdToNodeId,
							int totalNumFv,	int maxNumSN, unsigned int *pParitionMarker){
	int gTid = GLOBAL_TID();
	if(gTid >= totalNumFv)//thread has nothing to mark 
		return;

	unsigned int insId = pFvToInsId[gTid];
	int nid = pInsIdToNodeId[insId];
	if(nid < 0)
		return;
	int buffId = nid % maxNumSN;
	int partitionId = pBuffId2PartitionId[buffId];
	pParitionMarker[gTid] = partitionId;
}

/**
 * @brief: count number of elements in each segment in the partition marker
 */
__global__ void PartitionHistogram(unsigned int *pPartitionMarker, unsigned int markerLen, unsigned int numParition,
								   unsigned int numEleEachThd, unsigned int totalNumThd, unsigned int *pHistogram_d){
	extern __shared__ unsigned int counters[];
	int gTid = GLOBAL_TID();
	unsigned int tid = threadIdx.x;
	for(int p = 0; p < numParition; p++){
		counters[tid * numParition + p] = 0;
	}
	if(gTid >= totalNumThd){//thread has nothing to do
		return;
	}
	for(int i = 0; i < numEleEachThd; i++){
		if(gTid * numEleEachThd + i >= markerLen){//no element to process
			return;
		}
		int pid = pPartitionMarker[gTid * numEleEachThd + i];
		counters[tid * numParition + pid]++;
	}
	//store counters to global memory
	for(int p = 0; p < numParition; p++){
		//counters of the same partition are consecutive
		pHistogram_d[p * totalNumThd + gTid] = counters[tid * numParition + p];
	}
}

/**
 * @brief: compute feature value start position
 */
__global__ void ComputeFvalueStartPosEachNode(unsigned int *pHistogram_d, unsigned int totalNumThd,
											  unsigned int numNode, long long *pFeaValueStartPosEachNode){
	unsigned int arrayStartPos = 0;//start position of this array (i.e. node)
	int i = 0;
	do{
		pFeaValueStartPosEachNode[i] = arrayStartPos;//feature value start position of each node
		i++;
		if(i >= numNode)
			break;
		arrayStartPos += pHistogram_d[i * totalNumThd - 1];
	}while(true);
}

/**
  * @brief: store gather indices
  */
__global__ void CollectGatherIdx(const unsigned int *pPartitionMarker, unsigned int markerLen,
								 const unsigned int *pHistogram_d, unsigned int numParition,
								 unsigned int numEleEachThd, unsigned int totalNumThd, unsigned int *pGatherIdx){
	int gTid = GLOBAL_TID();
	if(gTid >= totalNumThd)//thread has nothing to collect
		return;

	unsigned int tid = threadIdx.x;
	extern __shared__ unsigned int eleDst[];

	//partition size
	for(int p = 0; p < numParition; p++){
		unsigned int partitionSize = pHistogram_d[p * totalNumThd + totalNumThd - 1];
		eleDst[tid * numParition + p] = partitionSize;
	}
	//partition start pos
	for(int p = 1; p < numParition; p++){
		eleDst[tid * numParition + p] += eleDst[tid * numParition + p - 1];
	}
	//write start pos
	for(int p = 0; p < numParition; p++){
		unsigned int thdCounterPos = p * totalNumThd + gTid;
		eleDst[tid * numParition + p] += pHistogram_d[thdCounterPos];
	}

	for(int i = 0; i < numEleEachThd; i++){
		unsigned int elePos = gTid * numEleEachThd + i;
		if(elePos >= markerLen)//no element to process
			return;
		int pid = pPartitionMarker[elePos];
		unsigned int writeIdx = tid * numParition + pid;
		pGatherIdx[elePos] = eleDst[writeIdx];//element destination
		eleDst[writeIdx]++;
	}
}


/**
  *@brief: compute length and start position of each feature in each node
  */
__global__ void ComputeEachFeaInfo(const long long *pEachFeaStartPos, int numFea, int totalNumFvalue, const unsigned int *pHistogram_d, int totalNumThd,
								   int *pEachFeaLenEachNode, long long *pEachFeaStartPosEachNode,
								   long long *pFvalueStartPosEachNode, long long *pNumFeaValueEachNode){
	int pid = blockIdx.x; //each arrayId corresponds to a prefix sum

	for(int f = 0; f < numFea; f++){
		unsigned int posOfLastFvalue;
		if(f < numFea - 1){
			PROCESS_ERROR(pEachFeaStartPos[f + 1] > 0);
			posOfLastFvalue = pid * totalNumFvalue + pEachFeaStartPos[f + 1] - 1;
		}
		else
			posOfLastFvalue = pid * totalNumFvalue + totalNumFvalue - 1;

		unsigned int startPos = pEachFeaStartPos[f];//start position of the feature f.
		unsigned int lenPreviousFvalue = 0;
		if(f > 0){
			lenPreviousFvalue = pHistogram_d[pid * totalNumThd + startPos - 1];
		}
		pEachFeaLenEachNode[pid * numFea + f] = pHistogram_d[posOfLastFvalue] - lenPreviousFvalue;
	}

	//start pos for first feature
	pEachFeaStartPosEachNode[pid * numFea] = pFvalueStartPosEachNode[pid];
	//start pos for other feature
	for(int f = 1; f < numFea; f++){
		unsigned int feaPos = pid * numFea + f;
		pEachFeaStartPosEachNode[feaPos] = pEachFeaStartPosEachNode[feaPos - 1] + pEachFeaLenEachNode[feaPos];
	}

	//number of feature values of this node
	pNumFeaValueEachNode[pid] = pHistogram_d[pid * totalNumThd + totalNumThd - 1];
}

/**
  * @brief: compute gether index by GPUs
  */
void IndexComputer::ComputeIdxGPU(int numSNode, int maxNumSN, int bagId){
	PROCESS_ERROR(m_totalFeaValue > 0 && numSNode > 0 && maxNumSN >= 0 && maxNumSN == m_maxNumofSN);
	
	int flags = -1;//all bits are 1
	BagManager bagManager;
	GBDTGPUMemManager manager;
	int *pBuffVec_d = bagManager.m_pBuffIdVecEachBag + bagId * bagManager.m_maxNumSplittable;

	//map snId to partition id start from 0
	int *pBuffVec_h = new int[numSNode];
	unsigned int *pBuffId2PartitionId_h = new unsigned int[m_maxNumofSN];
	checkCudaErrors(cudaMemcpy(pBuffVec_h, pBuffVec_d, sizeof(int) * numSNode, cudaMemcpyDeviceToHost));
	memset(pBuffId2PartitionId_h, flags, m_maxNumofSN);
	for(int i = 0; i < numSNode; i++){
		int snId = pBuffVec_h[i];
		pBuffId2PartitionId_h[snId] = i;
	}
	unsigned int *pBuffId2PartitionId_d;
	checkCudaErrors(cudaMalloc((void**)&pBuffId2PartitionId_d, sizeof(unsigned int) * m_maxNumofSN));
	checkCudaErrors(cudaMemcpy(pBuffId2PartitionId_d, pBuffId2PartitionId_h, sizeof(unsigned int) * m_maxNumofSN, cudaMemcpyHostToDevice));

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

	int *pTmpInsIdToNodeId = bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns;
	MarkPartition<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(pBuffId2PartitionId_d, manager.m_pDInsId, pTmpInsIdToNodeId,
																  m_totalFeaValue, maxNumSN, pPartitionMarker);
	GETERROR("after MarkPartition");

	unsigned int *pHistogram_d;
	unsigned int numElementEachThd = 16;
	unsigned int totalNumEffectiveThd = Ceil(m_totalFeaValue, numElementEachThd);
	dim3 numBlkDim;
	int numThdPerBlk;
	conf.ConfKernel(totalNumEffectiveThd, numThdPerBlk, numBlkDim);
	checkCudaErrors(cudaMalloc((void**)&pHistogram_d, sizeof(unsigned int) * numSNode * totalNumEffectiveThd));
	PartitionHistogram<<<numBlkDim, numThdPerBlk, numSNode * numThdPerBlk>>>(pPartitionMarker, m_totalFeaValue, numSNode,
																	     	 numElementEachThd, totalNumEffectiveThd, pHistogram_d);

	checkCudaErrors(cudaMalloc((void**)&pnKey, sizeof(unsigned int) * totalNumEffectiveThd * numSNode));
	for(int i = 0; i < numSNode; i++){
		int flag = (i % 2 == 0 ? 0:(-1));
		checkCudaErrors(cudaMemset(pnKey + i * totalNumEffectiveThd, flag, sizeof(unsigned int) * totalNumEffectiveThd));
	}
	//compute prefix sum for one array
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey, pnKey + totalNumEffectiveThd * numSNode,
								  pHistogram_d, pHistogram_d);//in place prefix sum

	//get feature values start position of each node
	long long *pTmpFvalueStartPosEachNode = bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;
	ComputeFvalueStartPosEachNode<<<1,1>>>(pHistogram_d, totalNumEffectiveThd, numSNode, pTmpFvalueStartPosEachNode);

	//write to gether index
	unsigned int *pTmpGatherIdx = bagManager.m_pIndicesEachBag_d + bagId * bagManager.m_numFeaValue;
	checkCudaErrors(cudaMemset(pTmpGatherIdx, flags, sizeof(unsigned int) * m_totalFeaValue));//when leaves appear, this is effective.
	CollectGatherIdx<<<numBlkDim, numThdPerBlk, numSNode * numThdPerBlk>>>(pPartitionMarker, m_totalFeaValue, pHistogram_d, numSNode,
												  numElementEachThd, totalNumEffectiveThd, pTmpGatherIdx);
	GETERROR("after CollectGatherIdx");

	//compute each feature length and start position in each node
	int *pTmpEachFeaLenEachNode = bagManager.m_pEachFeaLenEachNodeEachBag_d +
								  bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	long long * pTmpEachFeaStartPosEachNode = bagManager.m_pEachFeaStartPosEachNodeEachBag_d +
											  bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	ComputeEachFeaInfo<<<numSNode, 1>>>(manager.m_pFeaStartPos, m_numFea, totalNumEffectiveThd, pHistogram_d,
										pTmpEachFeaLenEachNode, pTmpEachFeaStartPosEachNode,
										pTmpFvalueStartPosEachNode, m_pNumFeaValueEachNode_dh);
	GETERROR("after ComputeEachFeaInfo");

	delete[] pBuffVec_h;
	delete[] pBuffId2PartitionId_h;
	checkCudaErrors(cudaFree(pBuffId2PartitionId_d));
	checkCudaErrors(cudaFree(pHistogram_d));
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

	checkCudaErrors(cudaMalloc((void**)&pPartitionMarker, sizeof(unsigned int) * m_totalFeaValue));
}
