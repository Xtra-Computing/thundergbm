/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "../pureHost/MyAssert.h"
#include "gbdtGPUMemManager.h"
#include "DeviceSplitter.h"
#include "DeviceSplitterKernel.h"

using std::cout;
using std::endl;


/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{

	int numofSNode = vBest.size();

	GBDTGPUMemManager manager;
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	//copy gradient and hessian to GPU memory
	float_point *pHostGD = new float_point[manager.m_numofIns];
	float_point *pHostHess = new float_point[manager.m_numofIns];
	for(int i = 0; i < manager.m_numofIns; i++)
	{
		pHostGD[i] = m_vGDPair_fixedPos[i].grad;
		pHostHess[i] = m_vGDPair_fixedPos[i].hess;
	}
	manager.MemcpyHostToDevice(pHostGD, manager.pGrad, sizeof(float_point) * manager.m_numofIns);
	manager.MemcpyHostToDevice(pHostHess, manager.pHess, sizeof(float_point) * manager.m_numofIns);
	float_point *pGD = manager.pGrad;
	float_point *pHess = manager.pHess;
	delete[] pHostGD;
	delete[] pHostHess;

	//copy instance id to node id infomation
	int *pInsToNodeId = new int[manager.m_numofIns];
	PROCESS_ERROR(manager.m_numofIns == m_nodeIds.size());
	manager.VecToArray(m_nodeIds, pInsToNodeId);
	manager.MemcpyHostToDevice(pInsToNodeId, manager.pInsIdToNodeId, sizeof(int) * manager.m_numofIns);
	delete[] pInsToNodeId;

	//copy splittable nodes to GPU memory
	int maxNumofSplittable = manager.m_maxNumofSplittable;
	int *pSNIdToBuffIdHost = new int[maxNumofSplittable];
	memset(pSNIdToBuffIdHost, -1, sizeof(int) * maxNumofSplittable);
	checkCudaErrors(cudaMemset(manager.pSNIdToBuffId, -1, sizeof(int) * maxNumofSplittable));
	nodeStat *pHostNodeStat = new nodeStat[maxNumofSplittable];
	int *pBuffId = new int[maxNumofSplittable];//save all the buffer ids
	int bidCounter = 0;
	for(map<int, int>::iterator it = mapNodeIdToBufferPos.begin(); it != mapNodeIdToBufferPos.end(); it++)
	{
		int snid = it->first;
		int vecId = it->second;
		PROCESS_ERROR(vecId < m_nodeStat.size());
		int buffId = AssignBufferId(pSNIdToBuffIdHost, snid, maxNumofSplittable);
		PROCESS_ERROR(buffId >= 0);
		pHostNodeStat[buffId] = m_nodeStat[vecId];

		//save the buffer ids into a set
		pBuffId[bidCounter] = buffId;
		bidCounter++;
	}

	manager.MemcpyHostToDevice(pHostNodeStat, manager.pSNodeStat, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyHostToDevice(pSNIdToBuffIdHost, manager.pSNIdToBuffId, sizeof(int) * maxNumofSplittable);

	//copy buffer ids to GPU memory
	PROCESS_ERROR(bidCounter == numofSNode);
	manager.MemcpyHostToDevice(pBuffId, manager.pBuffIdVec, sizeof(int) * maxNumofSplittable);
	nodeStat *pSNodeState = manager.pSNodeStat;

	delete[] pHostNodeStat;
	delete[] pBuffId;

	//use short names for temporary info
	nodeStat *pTempRChildStat = manager.pTempRChildStat;
	float_point *pLastValue = manager.pLastValue;

	//use short names for instance info
	int *pInsId = manager.pDInsId;
	float_point *pFeaValue = manager.pdDFeaValue;
	int *pNumofKeyValue = manager.pDNumofKeyValue;

	//reset the best splittable points
	SplitPoint *pBestPointHost = new SplitPoint[maxNumofSplittable];
	manager.MemcpyHostToDevice(pBestPointHost, manager.pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);
	delete []pBestPointHost;


	for(int f = 0; f < nNumofFeature; f++)
	{
		//the number of key values of the f{th} feature
		int numofCurFeaKeyValues = 0;
		manager.MemcpyDeviceToHost(pNumofKeyValue + f, &numofCurFeaKeyValues, sizeof(int));
		PROCESS_ERROR(numofCurFeaKeyValues > 0);

		long long startPosOfPrevFea = 0;
		int numofPreFeaKeyValues = 0;
		if(f > 0)
		{
			//number of key values of the previous feature
			manager.MemcpyDeviceToHost(pNumofKeyValue + (f - 1), &numofPreFeaKeyValues, sizeof(int));
			PROCESS_ERROR(numofPreFeaKeyValues > 0);
			//copy value of the start position of the previous feature
			manager.MemcpyDeviceToHost(manager.pFeaStartPos + (f - 1), &startPosOfPrevFea, sizeof(long long));
		}
		PROCESS_ERROR(startPosOfPrevFea >= 0);
		long long startPosOfCurFea = startPosOfPrevFea + numofPreFeaKeyValues;
		//copy the value of the start position of the current feature
		manager.MemcpyHostToDevice(&startPosOfCurFea, manager.pFeaStartPos + f, sizeof(long long));

		//reset the temporary right child statistics
		checkCudaErrors(cudaMemset(pTempRChildStat, 0, sizeof(nodeStat) * maxNumofSplittable));


		//find the split value for this feature
	int *idStartAddress = pInsId + startPosOfCurFea;
	float_point *pValueStartAddress = pFeaValue + startPosOfCurFea;

		FindFeaSplitValue<<<1, 1>>>(numofCurFeaKeyValues, idStartAddress, pValueStartAddress, manager.pInsIdToNodeId,
									pTempRChildStat, pGD, pHess, pLastValue, pSNodeState, manager.pBestSplitPoint,
									manager.pRChildStat, manager.pLChildStat, manager.pSNIdToBuffId,
									manager.m_maxNumofSplittable, f, manager.pBuffIdVec, numofSNode, DeviceSplitter::m_labda);
		cudaDeviceSynchronize();


		//copy back the best split points to vectors
		SplitPoint *pBestHost = new SplitPoint[maxNumofSplittable];
		nodeStat *pRChildStatHost = new nodeStat[maxNumofSplittable];
		nodeStat *pLChildStatHost = new nodeStat[maxNumofSplittable];
		manager.MemcpyDeviceToHost(manager.pBestSplitPoint, pBestHost, sizeof(SplitPoint) * maxNumofSplittable);
		manager.MemcpyDeviceToHost(manager.pRChildStat, pRChildStatHost, sizeof(nodeStat) * maxNumofSplittable);
		manager.MemcpyDeviceToHost(manager.pLChildStat, pLChildStatHost, sizeof(nodeStat) * maxNumofSplittable);
		for(map<int, int>::iterator it = mapNodeIdToBufferPos.begin(); it != mapNodeIdToBufferPos.end(); it++)
		{
			int snid = it->first;//splittable node id
			int vecId = it->second;//position id in vectors
			PROCESS_ERROR(vecId < m_nodeStat.size());
			//position id in buffer
			int buffId = GetBufferId(pSNIdToBuffIdHost, snid, maxNumofSplittable);
			PROCESS_ERROR(buffId >= 0);

			vBest[vecId] = pBestHost[buffId];
			rchildStat[vecId] = pRChildStatHost[buffId];
			lchildStat[vecId] = pLChildStatHost[buffId];
		}
		delete[] pBestHost;
		delete[] pRChildStatHost;
		delete[] pLChildStatHost;
	}

	delete[] pSNIdToBuffIdHost;//use twice
}

