/*
 * Preparator.cpp
 *
 *  Created on: 11 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "Preparator.h"
#include "Memory/gbdtGPUMemManager.h"
#include "../pureHost/MyAssert.h"
#include "Splitter/DeviceSplitter.h"
#include "Hashing.h"

int *DataPreparator::m_pSNIdToBuffIdHost = NULL;
int *DataPreparator::m_pUsedFIDMap = NULL;

/**
 * @brief: copy the gradient and hessian to GPU memory
 */
void DataPreparator::PrepareGDHess(const vector<gdpair> &m_vGDPair_fixedPos)
{
	GBDTGPUMemManager manager;

	//copy gradient and hessian to GPU memory
	float_point *pHostGD = new float_point[manager.m_numofIns];
	float_point *pHostHess = new float_point[manager.m_numofIns];
	for(int i = 0; i < manager.m_numofIns; i++)
	{
		pHostGD[i] = m_vGDPair_fixedPos[i].grad;
		pHostHess[i] = m_vGDPair_fixedPos[i].hess;
	}
	manager.MemcpyHostToDevice(pHostGD, manager.m_pGrad, sizeof(float_point) * manager.m_numofIns);
	manager.MemcpyHostToDevice(pHostHess, manager.m_pHess, sizeof(float_point) * manager.m_numofIns);
	delete[] pHostGD;
	delete[] pHostHess;
}

/**
 * @brief: copy splittable node information to GPU memory
 */
void DataPreparator::PrepareSNodeInfo(const map<int, int> &mapNodeIdToBufferPos, const vector<nodeStat> &m_nodeStat)
{
	GBDTGPUMemManager manager;
	//copy splittable nodes to GPU memory
	int maxNumofSplittable = manager.m_maxNumofSplittable;
	m_pSNIdToBuffIdHost = new int[maxNumofSplittable];
	memset(m_pSNIdToBuffIdHost, -1, sizeof(int) * maxNumofSplittable);
	nodeStat *pHostNodeStat = new nodeStat[maxNumofSplittable];
	int *pBuffId = new int[maxNumofSplittable];//save all the buffer ids
	int bidCounter = 0;
	for(map<int, int>::const_iterator it = mapNodeIdToBufferPos.begin(); it != mapNodeIdToBufferPos.end(); it++)
	{
		int snid = it->first;
		int vecId = it->second;
		PROCESS_ERROR(vecId < m_nodeStat.size());
		bool bIsNew = false;
		int buffId = Hashing::HostAssignHashValue(m_pSNIdToBuffIdHost, snid, maxNumofSplittable, bIsNew);
		PROCESS_ERROR(buffId >= 0);
		pHostNodeStat[buffId] = m_nodeStat[vecId];

		//save the buffer ids into a set
		pBuffId[bidCounter] = buffId;
		bidCounter++;
	}
	PROCESS_ERROR(bidCounter == m_nodeStat.size());

	if(mapNodeIdToBufferPos.size() > 1)//after first round; testing
	{
		//Splittable node ids to buffer ids
		int *pSNIDToBuffIdHost2 = new int[maxNumofSplittable];
		manager.MemcpyDeviceToHost(manager.m_pSNIdToBuffId, pSNIDToBuffIdHost2, sizeof(int) * maxNumofSplittable);
		for(int i = 0; i < maxNumofSplittable; i++)
		{
			//cout << "size=" << mapNodeIdToBufferPos.size() << "; " << pSNIDToBuffIdHost2[i] << " v.s. " << m_pSNIdToBuffIdHost[i] << endl;
			PROCESS_ERROR(pSNIDToBuffIdHost2[i] == m_pSNIdToBuffIdHost[i]);
		}
		delete[] pSNIDToBuffIdHost2;

		//number buffer ids
		int *pBuffIdHost = new int[maxNumofSplittable];
		manager.MemcpyDeviceToHost(manager.m_pBuffIdVec, pBuffIdHost, sizeof(int) * maxNumofSplittable);
		for(int i = 0; i < bidCounter; i++)
		{
			//cout << pBuffIdHost[i] << " v.s. " << pBuffId[i] << endl;
			PROCESS_ERROR(pBuffIdHost[i] == pBuffId[i]);
		}
	}
	else//first round
	{
		manager.Memset(manager.m_pSNIdToBuffId, -1, sizeof(int) * maxNumofSplittable);
		manager.MemcpyHostToDevice(m_pSNIdToBuffIdHost, manager.m_pSNIdToBuffId, sizeof(int) * maxNumofSplittable);
		manager.MemcpyHostToDevice(pHostNodeStat, manager.m_pSNodeStat, sizeof(nodeStat) * maxNumofSplittable);
		manager.MemcpyHostToDevice(pBuffId, manager.m_pBuffIdVec, sizeof(int) * maxNumofSplittable);
	}

	//copy buffer ids to GPU memory

	delete[] pHostNodeStat;
	delete[] pBuffId;
}

void DataPreparator::CopyBestSplitPoint(const map<int, int> &mapNodeIdToBufferPos, vector<SplitPoint> &vBest,
										vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{
	GBDTGPUMemManager manager;
	int maxNumofSplittable = manager.m_maxNumofSplittable;
	//copy back the best split points to vectors
	SplitPoint *pBestHost = new SplitPoint[maxNumofSplittable];
	nodeStat *pRChildStatHost = new nodeStat[maxNumofSplittable];
	nodeStat *pLChildStatHost = new nodeStat[maxNumofSplittable];
	manager.MemcpyDeviceToHost(manager.m_pBestSplitPoint, pBestHost, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pRChildStat, pRChildStatHost, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToHost(manager.m_pLChildStat, pLChildStatHost, sizeof(nodeStat) * maxNumofSplittable);
	for(map<int, int>::const_iterator it = mapNodeIdToBufferPos.begin(); it != mapNodeIdToBufferPos.end(); it++)
	{
		int snid = it->first;//splittable node id
		int vecId = it->second;//position id in vectors
		//position id in buffer
		int buffId = Hashing::HostGetBufferId(m_pSNIdToBuffIdHost, snid, maxNumofSplittable);
		PROCESS_ERROR(buffId >= 0);

		vBest[vecId] = pBestHost[buffId];
		rchildStat[vecId] = pRChildStatHost[buffId];
		lchildStat[vecId] = pLChildStatHost[buffId];
	}
	delete[] pBestHost;
	delete[] pRChildStatHost;
	delete[] pLChildStatHost;
}



