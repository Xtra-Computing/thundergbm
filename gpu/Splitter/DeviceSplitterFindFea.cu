/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "../../pureHost/MyAssert.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "DeviceSplitter.h"
#include "DeviceSplitterKernel.h"
#include "../Preparator.h"

using std::cout;
using std::endl;


/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{

	int numofSNode = vBest.size();

	GBDTGPUMemManager manager;
	//allocate reusable host memory
	manager.allocHostMemory();

	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	DataPreparator preparator;
	//copy gd and hess to GPU memory
	preparator.PrepareGDHess(m_vGDPair_fixedPos);

	float_point *pGD = manager.m_pGrad;
	float_point *pHess = manager.m_pHess;

	//copy instance id to node id infomation
	PROCESS_ERROR(manager.m_numofIns == m_nodeIds.size());
	manager.VecToArray(m_nodeIds, manager.m_pInsToNodeId);
	manager.MemcpyHostToDevice(manager.m_pInsToNodeId, manager.m_pInsIdToNodeId, sizeof(int) * manager.m_numofIns);

	//copy splittable node information and buffer ids to GPU memory
	preparator.PrepareSNodeInfo(mapNodeIdToBufferPos, m_nodeStat);
	nodeStat *pSNodeState = manager.m_pSNodeStat;

	//use short names for temporary info
	nodeStat *pTempRChildStat = manager.m_pTempRChildStat;
	float_point *pLastValue = manager.m_pLastValue;

	//use short names for instance info
	int *pInsId = manager.m_pDInsId;
	float_point *pFeaValue = manager.m_pdDFeaValue;
	int *pNumofKeyValue = manager.m_pDNumofKeyValue;

	//reset the best splittable points
	int maxNumofSplittable = manager.m_maxNumofSplittable;
	manager.MemcpyHostToDevice(manager.m_pBestPointHost, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);

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
			manager.MemcpyDeviceToHost(manager.m_pFeaStartPos + (f - 1), &startPosOfPrevFea, sizeof(long long));
		}
		PROCESS_ERROR(startPosOfPrevFea >= 0);
		long long startPosOfCurFea = startPosOfPrevFea + numofPreFeaKeyValues;
		//copy the value of the start position of the current feature
		manager.MemcpyHostToDevice(&startPosOfCurFea, manager.m_pFeaStartPos + f, sizeof(long long));

		//reset the temporary right child statistics
		checkCudaErrors(cudaMemset(pTempRChildStat, 0, sizeof(nodeStat) * maxNumofSplittable));


		//find the split value for this feature
		int *idStartAddress = pInsId + startPosOfCurFea;
		float_point *pValueStartAddress = pFeaValue + startPosOfCurFea;

		FindFeaSplitValue<<<1, 1>>>(numofCurFeaKeyValues, idStartAddress, pValueStartAddress, manager.m_pInsIdToNodeId,
									pTempRChildStat, pGD, pHess, pLastValue, pSNodeState, manager.m_pBestSplitPoint,
									manager.m_pRChildStat, manager.m_pLChildStat, manager.m_pSNIdToBuffId,
									manager.m_maxNumofSplittable, f, manager.m_pBuffIdVec, numofSNode, DeviceSplitter::m_labda);
		cudaDeviceSynchronize();


		//copy back the best split points to vectors
		preparator.CopyBestSplitPoint(mapNodeIdToBufferPos, vBest, rchildStat, lchildStat);
	}

	preparator.ReleaseMem();
}

int DeviceSplitter::AssignBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable)
{
	int buffId = -1;

	int remain = snid % m_maxNumofSplittable;//use mode operation as Hash function to find the buffer position

	//finding a location for snid
	if(pSNIdToBuffId[remain] == -1)
	{
		buffId = remain;
		pSNIdToBuffId[remain] = snid;
	}
	else
	{
		//Hash conflict
		for(int i = m_maxNumofSplittable - 1; i > 0; i--)
		{
			if(pSNIdToBuffId[i] == -1)
				buffId = i;
		}
	}

	return buffId;
}

/**
 * @brief: has an identical verion in device
 */
int DeviceSplitter::GetBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable)
{
	int buffId = -1;

	int remain = snid % m_maxNumofSplittable;//use mode operation as Hash function to find the buffer position

	//checking where snid is located
	if(pSNIdToBuffId[remain] == snid)
	{
		buffId = remain;
	}
	else
	{
		//Hash conflict
		for(int i = m_maxNumofSplittable - 1; i > 0; i--)
		{
			if(pSNIdToBuffId[i] == snid)
				buffId = i;
		}
	}

	return buffId;
}

