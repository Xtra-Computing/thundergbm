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

/**
 * @brief: compute index in new node for each feature value
 */
void IndexComputer::ComputeIndex(int numSNode, const int *pSNIdToBuffId, int maxNumSN, const int *pBuffVec)
{
//	PROCESS_ERROR(m_pInsId != NULL && m_totalFeaValue > 0 && m_insIdToNodeId_dh != NULL);
//	PROCESS_ERROR(numSNode > 0 && m_pIndices_dh != NULL);
//	PROCESS_ERROR(maxNumSN >= 0);

	//initialise length of each feature in each node
	memset(m_pEachFeaLenEachNode_dh, 0, sizeof(int) * m_numFea * m_maxNumofSN);

	//construct a mapping
	for(int b = 0; b < numSNode; b++)
	{
		int buffId = pBuffVec[b];
		m_pBuffIdToPos_dh[buffId] = b;
		m_pNumFeaValueEachNode_dh[b] = 0;//initialise the number of feature values of each node to 0
	}

	//compute fea value info for each node
	clock_t start_nodeFeaValue = clock();
	for(int fv = 0; fv < m_totalFeaValue; fv++)
	{
		int insId = m_pInsId[fv];
		int nid = m_insIdToNodeId_dh[insId];
		if(nid < 0)
		{
			continue;
		}
		int buffId = nid % maxNumSN;//pSNIdToBuffId[remain];//Hashing::HostGetBufferId(pSNIdToBuffId, nid, maxNumSN);
		int densePos = m_pBuffIdToPos_dh[buffId];

		//increase the number of fea values of this node by 1
		m_pNumFeaValueEachNode_dh[densePos]++;
	}
	clock_t end_nodeFeaValue = clock();
	m_total_copy += (end_nodeFeaValue - start_nodeFeaValue);

	//compute fea value start pos of each node
	for(int n = 0; n < numSNode; n++)
	{
		if(n == 0)
		{
			m_pFeaValueStartPosEachNode_dh[n] = 0;
		}
		else
		{
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

				for(int n = 0; n < numSNode; n++)
				{//initialise each feature start position and length
					m_pEachFeaStartPosEachNode_dh[feaId + n * m_numFea] = m_pIndexCounterEachNode[n];
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
		int snDensePos = m_pBuffIdToPos_dh[buffId];

		//compute index in the out array
		m_pIndices_dh[fv] = m_pIndexCounterEachNode[snDensePos];
//		PROCESS_ERROR(pIndexCounter[snDensePos] - m_pFeaValueStartPosEachNode_dh[snDensePos] < m_pNumFeaValueEachNode_dh[snDensePos]);

		m_pIndexCounterEachNode[snDensePos]++;//increase the index counter
		m_pEachFeaLenEachNode_dh[feaId + snDensePos * m_numFea]++;//increase the feature value length
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
}
