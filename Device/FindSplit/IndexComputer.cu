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
#include "../Hashing.h"
#include "../KernelConf.h"
#include "../prefix-sum/prefixSum.h"


using std::vector;

int *IndexComputer::m_pInsId = NULL;	//instance id for each feature value in the feature lists
int IndexComputer::m_totalFeaValue = -1;//total number of feature values in the whole dataset
long long *IndexComputer::m_pFeaStartPos = NULL;//each feature start position
int IndexComputer::m_numFea = -1;	//number of features
int IndexComputer::m_maxNumofSN = -1;
int *IndexComputer::m_pIndexCounterEachNode = NULL;
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

/**
  *@brief: mark feature values beloning to node with id=snId by 1
  */
__global__ void ArrayMarker(int snId, int *pFvToInsId, int *pInsIdToNodeId, int totalNumFv, int maxNumSN, float_point *pSparseGatherIdx){
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= totalNumFv)//thread has nothing to mark 
		return;

	int insId = pFvToInsId[gTid];
	int nid = pInsIdToNodeId[insId];
	int buffId = nid % maxNumSN;
	if(snId == buffId){
		pSparseGatherIdx[gTid] = 1;
	}
}

/**
  *@brief: compute length for each feature value of each node 
  *
  */
void UpdateEachFeaLenEachNode(unsigned int *pEachFeaStartPos, int snId, int numFea, int totalFvalue, unsigned int *pSparseGatherIdx, int *pEachFeaLenEachNode){
	for(int f = 0; f < numFea; f++){
		unsigned int posOfLastFvalue;
		if(f < numFea + 1)
			posOfLastFvalue = pEachFeaStartPos[f + 1] - 1;
		else
			posOfLastFvalue = totalFvalue - 1;

		unsigned int startPos = pEachFeaStartPos[f];
		unsigned int lenPreviousFvalue = 0;
		if(f > 0){
			lenPreviousFvalue = pSparseGatherIdx[startPos - 1];
		}
		pEachFeaLenEachNode[snId * numFea + f] = pSparseGatherIdx[posOfLastFvalue] - lenPreviousFvalue;
	}
}

/**
  * @brief: store gather indices
  */
__global__ void CollectGatherIdx(unsigned int *pSparseGatherIdx, unsigned int collectedGatherIdx, unsigned int totalNumFv, unsigned int *pGatherIdx){
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= totalNumFv)//thread has nothing to mark 
		return;

	unsigned int idx = pSparseGatherIdx[gTid];	
	if(gTid == 0){
		if(idx == 1)//store the first element
			pGatherIdx[idx + collectedGatherIdx - 1] = gTid;
		if(idx > 1 || idx < 0)
			printf("error in CollectGatherIdx\n");
	}
	else{
		if(idx == pSparseGatherIdx[gTid - 1])//repeated element due to prefix sum
			return;
		pGatherIdx[idx + collectedGatherIdx - 1] = gTid;
	}
}

__global__ void FloatToUnsignedInt(float_point *pfSparseGatherIdx, unsigned int totalNumFv, unsigned int *pnSparseGatherIdx){
	int gTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(gTid >= totalNumFv)//thread has nothing to mark 
		return;
	pnSparseGatherIdx[gTid] = pfSparseGatherIdx[gTid];
}


/**
  *@brief: compute start position of each feature in each node
  */
void ComputeEachFeaStartPosEachNode(int numFea, int snId, unsigned int collectedGatherIdx, int *pEachFeaLenEachNode, long long *pEachFeaStartPosEachNode){
	
	pEachFeaStartPosEachNode[snId * numFea] = collectedGatherIdx;
	for(int f = 1; f < numFea; f++){
		unsigned int feaPos = snId * numFea + f;
		pEachFeaStartPosEachNode[feaPos] = pEachFeaStartPosEachNode[feaPos - 1] + pEachFeaLenEachNode[feaPos];
	}
}

/**
  * @brief: compute gether index by GPUs
  */
void IndexComputer::ComputeIdxGPU(int numSNode, int maxNumSN, const int *pBuffVec){
	PROCESS_ERROR(m_pInsId != NULL && m_totalFeaValue > 0 && m_insIdToNodeId_dh != NULL);
	PROCESS_ERROR(numSNode > 0 && m_pIndices_dh != NULL);
	PROCESS_ERROR(maxNumSN >= 0);
	
	//this will be moved to memeory allocator
	float_point *pfSparseGatherIdx;
	unsigned int *pnSparseGatherIdx;
	checkCudaErrors(cudaMalloc((void**)&pfSparseGatherIdx, sizeof(float_point) * m_totalFeaValue));
	checkCudaErrors(cudaMalloc((void**)&pnSparseGatherIdx, sizeof(unsigned int) * m_totalFeaValue));

	unsigned int curGatherIdx = 0;
	unsigned int collectedGatherIdx = 0;

	KernelConf conf;
	int blockSizeForFvalue;
	dim3 dimNumofBlockForFvalue;
	conf.ConfKernel(m_totalFeaValue, blockSizeForFvalue, dimNumofBlockForFvalue);

	for(int i = 0; i < numSNode; i++){
		//reset sparse gather index
		checkCudaErrors(cudaMemset(pfSparseGatherIdx, 0, sizeof(float_point) * m_totalFeaValue));

		//construct 01 array for key values
		int snId = pBuffVec[i];//snId = nid % maxSN
		PROCESS_ERROR(snId >= 0);
		ArrayMarker<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(snId, m_pInsId, m_insIdToNodeId_dh, m_totalFeaValue, maxNumSN, pfSparseGatherIdx);

		//compute prefix sum for one array
		long long pnStartPos = 0;
		int arraySize = m_totalFeaValue;
		prefixsumForDeviceArray(pfSparseGatherIdx, &pnStartPos, &arraySize, 1, m_totalFeaValue);
		//convert float to int
		FloatToUnsignedInt<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(pfSparseGatherIdx, m_totalFeaValue, pnSparseGatherIdx);

		//compute each feature length in each node
		UpdateEachFeaLenEachNode(m_pEachFeaStartPos_dh, snId, m_numFea, m_totalFeaValue, pnSparseGatherIdx, m_pEachFeaLenEachNode_dh);

		//get collected gather index of this round
		checkCudaErrors(cudaMemcpy(&curGatherIdx, pnSparseGatherIdx + m_totalFeaValue - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));

		//write to gether index
		CollectGatherIdx<<<dimNumofBlockForFvalue, blockSizeForFvalue>>>(pnSparseGatherIdx, collectedGatherIdx, m_totalFeaValue, m_pnGatherIdx);

		//number of feature values of this node
		m_pNumFeaValueEachNode_dh[i] = curGatherIdx;//############# snId = i

		//each feature start position in each node
		ComputeEachFeaStartPosEachNode(m_numFea, snId, collectedGatherIdx, m_pEachFeaLenEachNode_dh, m_pEachFeaStartPosEachNode_dh);

		//feature value start position of each node
		m_pFeaValueStartPosEachNode_dh[i] = collectedGatherIdx;

		//update the number of collected gather indices
		collectedGatherIdx += curGatherIdx;
	}

	checkCudaErrors(cudaFree(pfSparseGatherIdx));
	checkCudaErrors(cudaFree(pnSparseGatherIdx));
}

/**
 * @brief: compute index in new node for each feature value
 * @output: scatter indices; len of each feature in each node; fvalue start pos of each node; start pos of each fea in each node
 */
void IndexComputer::ComputeIndex(int numSNode, const int *pSNIdToBuffId, int maxNumSN, const int *pBuffVec)
{
	PROCESS_ERROR(m_pInsId != NULL && m_totalFeaValue > 0 && m_insIdToNodeId_dh != NULL);
	PROCESS_ERROR(numSNode > 0 && m_pIndices_dh != NULL);
	PROCESS_ERROR(maxNumSN >= 0);

	//initialise length of each feature in each node
	memset(m_pEachFeaLenEachNode_dh, 0, sizeof(int) * m_numFea * m_maxNumofSN);

	//construct a mapping: buffId->snId. snId->buffId already exists
	for(int b = 0; b < numSNode; b++)
	{
		int buffId = pBuffVec[b];
		PROCESS_ERROR(buffId >= 0);
		m_pBuffIdToPos_dh[buffId] = b;
		m_pNumFeaValueEachNode_dh[b] = 0;//initialise the number of feature values of each node to 0
	}

	/*** compute fea value info for each node ***/
	clock_t start_nodeFeaValue = clock();
	printf("total feature value=%d\n", m_totalFeaValue);
	for(int fv = 0; fv < m_totalFeaValue; fv++)
	{
		int insId = m_pInsId[fv];
		int nid = m_insIdToNodeId_dh[insId];
		if(nid < 0){
			continue;
		}
		int buffId = nid % maxNumSN;//pSNIdToBuffId[remain];//Hashing::HostGetBufferId(pSNIdToBuffId, nid, maxNumSN);
		int snId = m_pBuffIdToPos_dh[buffId];
		PROCESS_ERROR(snId >= 0);

		//increase the number of fea values of this node by 1
		m_pNumFeaValueEachNode_dh[snId]++;
	}
	clock_t end_nodeFeaValue = clock();
	m_total_copy += (end_nodeFeaValue - start_nodeFeaValue);
	//compute fea value start pos of each node
	for(int n = 0; n < numSNode; n++){
		if(n == 0){
			m_pFeaValueStartPosEachNode_dh[n] = 0;
		}
		else{
			m_pFeaValueStartPosEachNode_dh[n] = m_pFeaValueStartPosEachNode_dh[n - 1] + m_pNumFeaValueEachNode_dh[n - 1];
		}
		//initialise start index
		m_pIndexCounterEachNode[n] = m_pFeaValueStartPosEachNode_dh[n];//counter for each node
	}

	//compute indices
	int feaId = -1;
	for(int fv = 0; fv < m_totalFeaValue; fv++)
	{
		if(m_numFea > feaId + 1 && fv == m_pFeaStartPos[feaId + 1])
		{
			clock_t start_copy = clock();
			do{
				feaId++;//next feature starts

				for(int n = 0; n < numSNode; n++){//initialise each feature start position and length
					m_pEachFeaStartPosEachNode_dh[feaId + n * m_numFea] = m_pIndexCounterEachNode[n];//m_pIndexCounterEachNode increases for each fv.
				}
			}while(m_numFea > feaId + 1 && fv == m_pFeaStartPos[feaId + 1]);//skip features having no values
			clock_t end_copy = clock();
			m_total_copy += (end_copy - start_copy);
		}

		int insId = m_pInsId[fv];
		int nid = m_insIdToNodeId_dh[insId];
		if(nid < 0)
		{
			//mark the position as dummy
			m_pIndices_dh[fv] = -1;
			continue;
		}
		int buffId = nid % maxNumSN;//pSNIdToBuffId[remain];//Hashing::HostGetBufferId(pSNIdToBuffId, nid, maxNumSN);
		int snId = m_pBuffIdToPos_dh[buffId];

		//compute index in the out array
		m_pIndices_dh[fv] = m_pIndexCounterEachNode[snId];
		PROCESS_ERROR(m_pIndexCounterEachNode[snId] - m_pFeaValueStartPosEachNode_dh[snId] < m_pNumFeaValueEachNode_dh[snId]);

		m_pIndexCounterEachNode[snId]++;//increase the index counter
		m_pEachFeaLenEachNode_dh[feaId + snId * m_numFea]++;//increase the feature value length
	}
}

/**
 * @brief: allocate reusable memory
 */
void IndexComputer::AllocMem(int nNumofExamples, int nNumofFeatures, int maxNumofSplittableNode)
{
	m_numFea = nNumofFeatures;
	m_maxNumofSN = maxNumofSplittableNode;

	m_pIndexCounterEachNode = new int[m_maxNumofSN];
	checkCudaErrors(cudaMallocHost((void**)&m_pIndices_dh, sizeof(int) * m_totalFeaValue));
	checkCudaErrors(cudaMallocHost((void**)&m_insIdToNodeId_dh, sizeof(int) * nNumofExamples));
	checkCudaErrors(cudaMallocHost((void**)&m_pNumFeaValueEachNode_dh, sizeof(long long) * m_maxNumofSN));
	checkCudaErrors(cudaMallocHost((void**)&m_pBuffIdToPos_dh, sizeof(int) * m_maxNumofSN));
	checkCudaErrors(cudaMallocHost((void**)&m_pFeaValueStartPosEachNode_dh, sizeof(long long) * m_maxNumofSN));
	checkCudaErrors(cudaMallocHost((void**)&m_pEachFeaStartPosEachNode_dh, sizeof(long long) * m_maxNumofSN * m_numFea));
	checkCudaErrors(cudaMallocHost((void**)&m_pEachFeaLenEachNode_dh, sizeof(int) * m_maxNumofSN * m_numFea));

	checkCudaErrors(cudaMallocHost((void**)&m_pEachFeaStartPos_dh, sizeof(unsigned int) * m_numFea));
	checkCudaErrors(cudaMalloc((void**)&m_pnGatherIdx, sizeof(unsigned int) * m_totalFeaValue));
}
