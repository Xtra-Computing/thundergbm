/*
 * IndexComputer.cpp
 *
 *  Created on: 21 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: compute index for each feature value in the feature lists
 */

#include <vector>
#include <algorithm>
#include "IndexComputer.h"
#include "../../DeviceHost/MyAssert.h"
#include "../Hashing.h"

using std::vector;

int *IndexComputer::m_pInsId = NULL;	//instance id for each feature value in the feature lists
int IndexComputer::m_totalFeaValue = -1;//total number of feature values in the whole dataset
long long *IndexComputer::m_pFeaStartPos = NULL;//each feature start position
int IndexComputer::m_numFea = -1;	//number of features

int *IndexComputer::m_insIdToNodeId_dh = NULL;//instance id to node id
int *IndexComputer::m_pIndices_dh = NULL;	//index for each node
int *IndexComputer::m_pNumFeaValueEachNode_dh = NULL;	//# of feature values of each node
int *IndexComputer::m_pFeaValueStartPosEachNode_dh = NULL;//start positions to feature value of each node
int *IndexComputer::m_pEachFeaStartPosEachNode_dh = NULL;//each feature start position in each node
int *IndexComputer::m_pEachFeaLenEachNode_dh = NULL;//each feature value length in each node
int *IndexComputer::m_pBuffIdToPos_dh = NULL;//map buff id to dense pos id; not all elements in this array are used, due to not continuous buffid.
int *IndexComputer::m_pPosToBuffId_dh = NULL;//map dense pos id to buff id


int IndexComputer::MapBuffIdToPos(int buffId, const int *pBuffIdToPos)
{
	int pos = pBuffIdToPos[buffId];
	PROCESS_ERROR(pos >= 0);
	return pos;
}

/**
 * @brief: compute index in new node for each feature value
 */
void IndexComputer::ComputeIndex(int numSNode, const int *pSNIdToBuffId, int maxNumSN, const int *pBuffVec)
{
	PROCESS_ERROR(m_pInsId != NULL && m_totalFeaValue > 0 && m_insIdToNodeId_dh != NULL);
	PROCESS_ERROR(numSNode > 0 && m_pIndices_dh != NULL);
	PROCESS_ERROR(maxNumSN >= 0);

	//buffId to continuous ids
	vector<int> buffIdToDensePos;
	for(int n = 0; n < numSNode; n++)
	{
		buffIdToDensePos.push_back(pBuffVec[n]);
	}
	sort(buffIdToDensePos.begin(), buffIdToDensePos.end());
	//construct a mapping

	for(int b = 0; b < numSNode; b++)
	{
		int truncatedBuffId = buffIdToDensePos[b];
		m_pBuffIdToPos_dh[truncatedBuffId] = b;
		m_pPosToBuffId_dh[b] = truncatedBuffId;
		m_pNumFeaValueEachNode_dh[b] = 0;//initialise the number of feature values of each node to 0
	}

	//compute fea value info for each node
	for(int fv = 0; fv < m_totalFeaValue; fv++)
	{
		int insId = m_pInsId[fv];
		int nid = m_insIdToNodeId_dh[insId];
		if(nid == -1)
		{
			continue;
		}
		int buffId = Hashing::HostGetBufferId(pSNIdToBuffId, nid, maxNumSN);
		int densePos = MapBuffIdToPos(buffId, m_pBuffIdToPos_dh);

		//increase the number of fea values of this node by 1
		m_pNumFeaValueEachNode_dh[densePos]++;
	}

	//create counter for each node
	int *pIndexCounter = new int[numSNode];
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
		pIndexCounter[n] = m_pFeaValueStartPosEachNode_dh[n];
	}

	//compute indices
	int feaId = -1;
	for(int fv = 0; fv < m_totalFeaValue; fv++)
	{
		if(m_numFea > feaId + 1 && fv == m_pFeaStartPos[feaId + 1])
		{
			feaId++;//next feature starts
			for(int n = 0; n < numSNode; n++)
			{//initialise each feature start position and length
				m_pEachFeaStartPosEachNode_dh[feaId + n * m_numFea] = pIndexCounter[n];
				m_pEachFeaLenEachNode_dh[feaId + n * m_numFea] = 0;
			}
		}

		int insId = m_pInsId[fv];
		int nid = m_insIdToNodeId_dh[insId];
		if(nid == -1)
		{
			//mark the position as dummy
			m_pIndices_dh[fv] = -1;
			continue;
		}
		int buffId = Hashing::HostGetBufferId(pSNIdToBuffId, nid, maxNumSN);
		int snDensePos = MapBuffIdToPos(buffId, m_pBuffIdToPos_dh);

		//compute index in the out array
		int index = pIndexCounter[snDensePos];
		m_pIndices_dh[fv] = index;
		PROCESS_ERROR(pIndexCounter[snDensePos] - m_pFeaValueStartPosEachNode_dh[snDensePos] < m_pNumFeaValueEachNode_dh[snDensePos]);

		pIndexCounter[snDensePos]++;//increase the index counter
		m_pEachFeaLenEachNode_dh[feaId + snDensePos * m_numFea]++;//increase the feature value length
	}

	delete[] pIndexCounter;
}
