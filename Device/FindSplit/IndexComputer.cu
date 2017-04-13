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
#include "../Bagging/BagManager.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../DeviceHost/MyAssert.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/powerOfTwo.h"
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

int* IndexComputer::m_pArrangedInsId_d = NULL;
float_point* IndexComputer::m_pArrangedFvalue_d = NULL;

/**
  *@brief: mark feature values beloning to node with id=snId by 1
  */
__global__ void MarkPartition(unsigned int *pSNPos2PartitionId, int *pFvToInsId, int *pInsIdToNodeId,
							int totalNumFv,	int maxNumSN, unsigned int *pParitionMarker){
	int gTid = GLOBAL_TID();
	if(gTid >= totalNumFv)//thread has nothing to mark 
		return;

	unsigned int insId = pFvToInsId[gTid];
	int nid = pInsIdToNodeId[insId];
	if(nid < 0)
		return;
	int snPos = nid % maxNumSN;
	int partitionId = pSNPos2PartitionId[snPos];
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
			break;
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
	extern __shared__ unsigned int eleDst[];//effectively, 4 counters for each thread

	//will be improved later############
	//right shifted partition size
	eleDst[tid * numParition] = 0;
	for(int p = 1; p < numParition; p++){
		unsigned int partitionSize = pHistogram_d[(p - 1) * totalNumThd + totalNumThd - 1];
		eleDst[tid * numParition + p] = partitionSize;
	}
	//partition start pos (i.e. prefix sum)
	for(int p = 1; p < numParition; p++){
		eleDst[tid * numParition + p] += eleDst[tid * numParition + p - 1];
	}
	//end--will be improved later########
	//write start pos of each thread
	for(int p = 0; p < numParition; p++){
		unsigned int thdCounterPos = p * totalNumThd + gTid;
		eleDst[tid * numParition + p] += pHistogram_d[thdCounterPos];//inclusive prefix sum in pHistgram_d
		unsigned int includedExtraValue;
		if(gTid == 0)
			includedExtraValue = pHistogram_d[thdCounterPos];
		else
			includedExtraValue = pHistogram_d[thdCounterPos] - pHistogram_d[thdCounterPos - 1];
		eleDst[tid * numParition + p] -= includedExtraValue;
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
__global__ void ComputeEachFeaInfo(const unsigned int *pPartitionMarker, const unsigned int *pGatherIdx, int totalNumFvalue,
								   const long long *pFvalueStartPosEachSN, int numFea,
								   const unsigned int *pHistogram_d, int totalNumThd,
								   int *pEachFeaLenEachNode, long long *pEachFeaStartPosEachNode,
								   long long *pNumFeaValueEachSN){
	int previousPid = threadIdx.x; //each thread corresponds to a splittable node
	extern __shared__ unsigned int eachFeaLenEachNewNode[];

	//get pids for this node
	int start = pFvalueStartPosEachSN[previousPid];
	int lenofSN = pNumFeaValueEachSN[previousPid];
	int pid1 = pPartitionMarker[start];
	int pid2 = -1;
	for(int i = 0; i < lenofSN; i++){
		int p = pPartitionMarker[start + i];
		if(p != pid1){
				pid2 = p;
				break;
		}
	}
	printf("pid1=%d v.s. pid2=%d, # of tid=%d\n", pid1, pid2, blockDim.x);

	//get pid1 and pid2 start position
	unsigned int startPosPartition1 = 0;
	unsigned int startPosPartition2 = 0;
	//partition pid1 start pos (i.e. prefix sum)
	for(int p = 0; p < pid1; p++){
		startPosPartition1 += pHistogram_d[p * totalNumThd + totalNumThd - 1];
	}
	//partition pid2 start pos (i.e. prefix sum)
	for(int p = 0; p < pid2; p++){
		startPosPartition2 += pHistogram_d[p * totalNumThd + totalNumThd - 1];
	}

	unsigned int startPosofCurFeaPid1 = startPosPartition1;
	unsigned int startPosofCurFeaPid2 = startPosPartition2;
	for(int f = 0; f < numFea; f++){
		//get lengths for f in the two partitions
		unsigned int feaPos = previousPid * numFea + f;
		unsigned int numFvalueThisSN = pEachFeaLenEachNode[feaPos];
		unsigned int posOfLastFValue = pEachFeaStartPosEachNode[feaPos] + numFvalueThisSN - 1;
		int lastFvaluePid = pPartitionMarker[posOfLastFValue];

		//get length of f in partition that contains last fvalue
		unsigned int dstPos = pGatherIdx[posOfLastFValue];
		unsigned int startPosofCurFea = 0;
		if(lastFvaluePid == pid1)
			startPosofCurFea = startPosofCurFeaPid1;
		else
			startPosofCurFea = startPosofCurFeaPid2;

		unsigned int numThisFeaValue = dstPos - startPosofCurFea + 1;

		//start position for the next feature
		if(lastFvaluePid == pid1){
			startPosofCurFeaPid1 += numThisFeaValue;
			startPosofCurFeaPid2 += (numFvalueThisSN - numThisFeaValue);
		}
		else{
			startPosofCurFeaPid2 += numThisFeaValue;
			startPosofCurFeaPid1 += (numFvalueThisSN - numThisFeaValue);
		}

		//temporarily store each feature length in shared memory
		if(lastFvaluePid == pid1){
			eachFeaLenEachNewNode[pid1 * numFea + f] = numThisFeaValue;
			eachFeaLenEachNewNode[pid2 * numFea + f] = (numFvalueThisSN - numThisFeaValue);
		}
		else{
			eachFeaLenEachNewNode[pid2 * numFea + f] = numThisFeaValue;
			eachFeaLenEachNewNode[pid1 * numFea + f] = (numFvalueThisSN - numThisFeaValue);
		}
	}

	__syncthreads();
	//update each fea len
	for(int f = 0; f < numFea; f++){
		pEachFeaLenEachNode[pid1 * numFea + f] = eachFeaLenEachNewNode[pid1 * numFea + f];
		pEachFeaLenEachNode[pid2 * numFea + f] = eachFeaLenEachNewNode[pid2 * numFea + f];
	}
	//start pos for first feature
	pEachFeaStartPosEachNode[pid1 * numFea] = startPosPartition1;
	pEachFeaStartPosEachNode[pid2 * numFea] = startPosPartition2;
	//start pos for other feature
	for(int f = 1; f < numFea; f++){
		unsigned int feaPosPid1 = pid1 * numFea + f;
		unsigned int feaPosPid2 = pid2 * numFea + f;
		pEachFeaStartPosEachNode[feaPosPid1] = pEachFeaStartPosEachNode[feaPosPid1 - 1] + pEachFeaLenEachNode[feaPosPid1];
		pEachFeaStartPosEachNode[feaPosPid2] = pEachFeaStartPosEachNode[feaPosPid2 - 1] + pEachFeaLenEachNode[feaPosPid2];
	}

	//update number of feature values of each new node
	pNumFeaValueEachSN[pid1] = pHistogram_d[pid1 * totalNumThd + totalNumThd - 1];
	pNumFeaValueEachSN[pid2] = pHistogram_d[pid2 * totalNumThd + totalNumThd - 1];
}

/**
  * @brief: compute gether index by GPUs
  */
void IndexComputer::ComputeIdxGPU(int numSNode, int maxNumSN, int bagId){
	PROCESS_ERROR(m_totalFeaValue > 0 && numSNode > 0 && maxNumSN >= 0 && maxNumSN == m_maxNumofSN);
	
	int flags = -1;//all bits are 1
	BagManager bagManager;
	int *pPartitionId2SNPos_d = bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable;

	//map snId to partition id start from 0
	int *pPartitionId2SNPos_h = new int[numSNode];
	unsigned int *pSNPos2PartitionId_h = new unsigned int[m_maxNumofSN];
	checkCudaErrors(cudaMemcpy(pPartitionId2SNPos_h, pPartitionId2SNPos_d, sizeof(int) * numSNode, cudaMemcpyDeviceToHost));
	memset(pSNPos2PartitionId_h, flags, m_maxNumofSN);
	for(int i = 0; i < numSNode; i++){
		int snPos = pPartitionId2SNPos_h[i];
		pSNPos2PartitionId_h[snPos] = i;
	}
	unsigned int *pSNPos2PartitionId_d;
	checkCudaErrors(cudaMalloc((void**)&pSNPos2PartitionId_d, sizeof(unsigned int) * m_maxNumofSN));
	checkCudaErrors(cudaMemcpy(pSNPos2PartitionId_d, pSNPos2PartitionId_h, sizeof(unsigned int) * m_maxNumofSN, cudaMemcpyHostToDevice));

	KernelConf conf;
	int blockSizeForFvalue;
	dim3 dimNumofBlockForFvalue;
	conf.ConfKernel(m_totalFeaValue, blockSizeForFvalue, dimNumofBlockForFvalue);
	if(dimNumofBlockForFvalue.z > 1){
		printf("invalid kernel configuration!\n");
		exit(0);
	}

	int *pTmpInsIdToNodeId = bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns;
	MarkPartition<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(pSNPos2PartitionId_d, m_pArrangedInsId_d, pTmpInsIdToNodeId,
																  m_totalFeaValue, maxNumSN, pPartitionMarker);
	GETERROR("after MarkPartition");

	unsigned int *pHistogram_d;
	unsigned int numElementEachThd = 16;
	unsigned int totalNumEffectiveThd = Ceil(m_totalFeaValue, numElementEachThd);
	dim3 numBlkDim;
	int numThdPerBlk;
	conf.ConfKernel(totalNumEffectiveThd, numThdPerBlk, numBlkDim);
	checkCudaErrors(cudaMalloc((void**)&pHistogram_d, sizeof(unsigned int) * numSNode * totalNumEffectiveThd));
	PartitionHistogram<<<numBlkDim, numThdPerBlk, numSNode * numThdPerBlk * sizeof(unsigned int)>>>(pPartitionMarker, m_totalFeaValue, numSNode,
																	     	 numElementEachThd, totalNumEffectiveThd, pHistogram_d);
	GETERROR("after PartitionHistogram");
	checkCudaErrors(cudaMalloc((void**)&pnKey, sizeof(unsigned int) * totalNumEffectiveThd * numSNode));
	for(int i = 0; i < numSNode; i++){
		int flag = (i % 2 == 0 ? 0:(-1));
		checkCudaErrors(cudaMemset(pnKey + i * totalNumEffectiveThd, flag, sizeof(unsigned int) * totalNumEffectiveThd));
	}
	//compute prefix sum for one array
	thrust::inclusive_scan_by_key(thrust::system::cuda::par, pnKey, pnKey + totalNumEffectiveThd * numSNode,
								  pHistogram_d, pHistogram_d);//in place prefix sum

	//write to gather index
	unsigned int *pTmpGatherIdx = bagManager.m_pIndicesEachBag_d + bagId * bagManager.m_numFeaValue;
	checkCudaErrors(cudaMemset(pTmpGatherIdx, flags, sizeof(unsigned int) * m_totalFeaValue));//when leaves appear, this is effective.
	CollectGatherIdx<<<numBlkDim, numThdPerBlk, numSNode * numThdPerBlk * sizeof(unsigned int)>>>(pPartitionMarker, m_totalFeaValue, pHistogram_d, numSNode,
												  numElementEachThd, totalNumEffectiveThd, pTmpGatherIdx);
	GETERROR("after CollectGatherIdx");

	long long *pTmpFvalueStartPosEachNode = bagManager.m_pFvalueStartPosEachNodeEachBag_d + bagId * bagManager.m_maxNumSplittable;
	//compute each feature length and start position in each node
	int *pTmpEachFeaLenEachNode = bagManager.m_pEachFeaLenEachNodeEachBag_d +
								  bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	long long * pTmpEachFeaStartPosEachNode = bagManager.m_pEachFeaStartPosEachNodeEachBag_d +
											  bagId * bagManager.m_maxNumSplittable * bagManager.m_numFea;
	int numThd = Ceil(numSNode, 2);
	ComputeEachFeaInfo<<<1, numThd, numSNode * m_numFea * sizeof(unsigned int)>>>(pPartitionMarker, pTmpGatherIdx, m_totalFeaValue,
										pTmpFvalueStartPosEachNode, m_numFea,
										pHistogram_d, totalNumEffectiveThd,
										pTmpEachFeaLenEachNode, pTmpEachFeaStartPosEachNode,
										 m_pNumFeaValueEachNode_dh);
	GETERROR("after ComputeEachFeaInfo");

	//get feature values start position of each new node
	ComputeFvalueStartPosEachNode<<<1,1>>>(pHistogram_d, totalNumEffectiveThd, numSNode, pTmpFvalueStartPosEachNode);

	delete[] pPartitionId2SNPos_h;
	delete[] pSNPos2PartitionId_h;
	checkCudaErrors(cudaFree(pSNPos2PartitionId_d));
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

	checkCudaErrors(cudaMalloc((void**)&m_pArrangedInsId_d, sizeof(int) * m_totalFeaValue));
	checkCudaErrors(cudaMalloc((void**)&m_pArrangedFvalue_d, sizeof(float_point) * m_totalFeaValue));
}
