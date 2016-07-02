/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "DeviceSplitter.h"
#include "DeviceFindFeaKernel.h"
#include "Initiator.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../Memory/SplitNodeMemManager.h"
#include "../Memory/dtMemManager.h"
#include "../Preparator.h"
#include "../Hashing.h"
#include "../DevicePredictorHelper.h"
#include "../DevicePredictor.h"
#include "../KernelConf.h"
#include "../../DeviceHost/MyAssert.h"
#include "../../DeviceHost/SparsePred/DenseInstance.h"

using std::cout;
using std::endl;
using std::make_pair;


/**
 * @brief: efficient best feature finder
 */
void DeviceSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{
	GBDTGPUMemManager manager;
	int numofSNode = manager.m_curNumofSplitable;
	int tempSN = 0;
	manager.MemcpyDeviceToHost(manager.m_pNumofBuffId, &tempSN, sizeof(int));
	PROCESS_ERROR(numofSNode == tempSN);
	int nNumofFeature = manager.m_numofFea;
	PROCESS_ERROR(nNumofFeature > 0);

	//gd and hess short name on GPU memory
	float_point *pGD = manager.m_pGrad;
	float_point *pHess = manager.m_pHess;

	//splittable node information short name on GPU memory
	nodeStat *pSNodeState = manager.m_pSNodeStat;

	//use short names for temporary info on GPU memory
	nodeStat *pTempRChildStat = manager.m_pTempRChildStat;
	float_point *pLastValue = manager.m_pLastValue;

	//use short names for instance info
	int *pInsId = manager.m_pDInsId;
	float_point *pFeaValue = manager.m_pdDFeaValue;
	int *pNumofKeyValue = manager.m_pDNumofKeyValue;

	int maxNumofSplittable = manager.m_maxNumofSplittable;
	//Memory set for best split points (i.e. reset the best splittable points)
	manager.MemcpyHostToDevice(manager.m_pBestPointHost, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);

	//allocate numofFeature*numofSplittabeNode
	manager.allocMemForSNForEachThread(nNumofFeature, manager.m_maxNumofSplittable);
	for(int f = 0; f < nNumofFeature; f++)
		manager.MemcpyDeviceToDevice(pSNodeState, manager.m_pSNodeStatPerThread + f * maxNumofSplittable, sizeof(nodeStat) * maxNumofSplittable);

	KernelConf conf;
	dim3 dimGridThreadForEachFea;
	conf.ComputeBlock(nNumofFeature, dimGridThreadForEachFea);
	int sharedMemSizeEachFea = 1;
	cudaDeviceSynchronize();
	FindFeaSplitValue<<<dimGridThreadForEachFea, sharedMemSizeEachFea>>>(
									  pNumofKeyValue, manager.m_pFeaStartPos, pInsId, pFeaValue, manager.m_pInsIdToNodeId,
									  pGD, pHess,
									  manager.m_pTempRChildStatPerThread, manager.m_pLastValuePerThread,
									  manager.m_pSNodeStatPerThread, manager.m_pBestSplitPointPerThread,
									  manager.m_pRChildStatPerThread, manager.m_pLChildStatPerThread,
									  manager.m_pSNIdToBuffId, maxNumofSplittable, manager.m_pBuffIdVec, numofSNode,
									  DeviceSplitter::m_lambda, nNumofFeature);
	cudaDeviceSynchronize();
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in FindFeaSplitValue" << endl;
		exit(0);
	}
#endif

	PickBestFea<<<1, 1>>>(manager.m_pLastValuePerThread,
						  manager.m_pBestSplitPointPerThread, manager.m_pRChildStatPerThread, manager.m_pLChildStatPerThread,
						  manager.m_pBuffIdVec, numofSNode, nNumofFeature, maxNumofSplittable);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in PickBestFea" << endl;
		exit(0);
	}
#endif

	manager.MemcpyDeviceToDevice(manager.m_pLastValuePerThread, manager.m_pLastValue, sizeof(float_point) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pBestSplitPointPerThread, manager.m_pBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pRChildStatPerThread, manager.m_pRChildStat, sizeof(nodeStat) * maxNumofSplittable);
	manager.MemcpyDeviceToDevice(manager.m_pLChildStatPerThread, manager.m_pLChildStat, sizeof(nodeStat) * maxNumofSplittable);

	manager.freeMemForSNForEachThread();
	//print best split points
#if false
	int *pTestBuffIdVect = new int[numofSNode];
	manager.MemcpyDeviceToHost(manager.m_pBuffIdVec, pTestBuffIdVect, sizeof(int) * numofSNode);
	SplitPoint *testBestSplitPoint = new SplitPoint[maxNumofSplittable];
	manager.MemcpyDeviceToHost(manager.m_pBestSplitPoint, testBestSplitPoint, sizeof(SplitPoint) * maxNumofSplittable);
	for(int sn = 0; sn < numofSNode; sn++)
	{
		cout << "nid=" << pTestBuffIdVect[sn] << "; snid=" << sn << "; gain=" << testBestSplitPoint[pTestBuffIdVect[sn]].m_fGain << "; fid="
			 << testBestSplitPoint[pTestBuffIdVect[sn]].m_nFeatureId << "; sv=" << testBestSplitPoint[pTestBuffIdVect[sn]].m_fSplitValue << endl;
	}
	delete[] testBestSplitPoint;
	delete[] pTestBuffIdVect;
#endif
}

/**
 * @brief: prediction and compute gradient descent
 */
void DeviceSplitter::ComputeGD(vector<RegTree> &vTree, vector<vector<KeyValue> > &vvInsSparse)
{
	GBDTGPUMemManager manager;
	DevicePredictor pred;
	//get features and store the feature ids in a way that the access is efficient
	DenseInsConverter denseInsConverter(vTree);

	//hash feature id to position id
	int numofUsedFea = denseInsConverter.usedFeaSet.size();
	int *pHashUsedFea = NULL;
	int *pSortedUsedFea = NULL;
	pred.GetUsedFeature(denseInsConverter.usedFeaSet, pHashUsedFea, pSortedUsedFea);

	//for each tree
	int nNumofTree = vTree.size();
	int nNumofIns = manager.m_numofIns;
	PROCESS_ERROR(nNumofIns > 0);

	//the last learned tree
	int numofNodeOfLastTree = 0;
	TreeNode *pLastTree = NULL;
	DTGPUMemManager treeManager;
	int numofTreeLearnt = treeManager.m_numofTreeLearnt;
	int treeId = numofTreeLearnt - 1;
	pred.GetTreeInfo(pLastTree, numofNodeOfLastTree, treeId);

	//start prediction
	checkCudaErrors(cudaMemset(manager.m_pTargetValue, 0, sizeof(float_point) * nNumofIns));
	if(nNumofTree > 0)
	{
		long long startPos = 0;
		int startInsId = 0;
		long long *pInsStartPos = manager.m_pInsStartPos + startInsId;
		manager.MemcpyDeviceToHost(pInsStartPos, &startPos, sizeof(long long));
	//			cout << "start pos ins" << insId << "=" << startPos << endl;
		float_point *pDevInsValue = manager.m_pdDInsValue + startPos;
		int *pDevFeaId = manager.m_pDFeaId + startPos;
		int *pNumofFea = manager.m_pDNumofFea + startInsId;
		int numofInsToFill = nNumofIns;

		FillMultiDense<<<numofInsToFill, 1>>>(pDevInsValue, pInsStartPos, pDevFeaId, pNumofFea, manager.m_pdDenseIns,
											  manager.m_pSortedUsedFeaId, manager.m_pHashFeaIdToDenseInsPos,
											  numofUsedFea, startInsId, numofInsToFill);
#if testing
			if(cudaGetLastError() != cudaSuccess)
			{
				cout << "error in FillMultiDense" << endl;
				exit(0);
			}
#endif
	}


	for(int i = 0; i < nNumofIns; i++)
	{
		if(nNumofTree > 0)
		{
//			pred.FillDenseIns(i, numofUsedFea);
			int denseInsStartPos = i * numofUsedFea;

			//prediction using the last tree
			PROCESS_ERROR(numofUsedFea <= manager.m_maxUsedFeaInTrees);
			assert(pLastTree != NULL);
			PredTarget<<<1, 1>>>(pLastTree, numofNodeOfLastTree, manager.m_pdDenseIns + denseInsStartPos, numofUsedFea,
								 manager.m_pHashFeaIdToDenseInsPos, manager.m_pTargetValue + i, treeManager.m_maxTreeDepth);
#if testing
			if(cudaGetLastError() != cudaSuccess)
			{
				cout << "error in PredTarget" << endl;
				exit(0);
			}
#endif
		}
		manager.MemcpyDeviceToDevice(manager.m_pTargetValue + i, manager.m_pPredBuffer + i, sizeof(float_point));
	}

	if(pHashUsedFea != NULL)
		delete []pHashUsedFea;
	if(pSortedUsedFea != NULL)
		delete []pSortedUsedFea;

	//compute GD
	ComputeGDKernel<<<1, 1>>>(nNumofIns, manager.m_pTargetValue, manager.m_pdTrueTargetValue, manager.m_pGrad, manager.m_pHess);

	//copy splittable nodes to GPU memory
		//SNodeStat, SNIdToBuffId, pBuffIdVec need to be reset.
	manager.Memset(manager.m_pSNodeStat, 0, sizeof(nodeStat) * manager.m_maxNumofSplittable);
	manager.Memset(manager.m_pSNIdToBuffId, -1, sizeof(int) * manager.m_maxNumofSplittable);
	manager.Memset(manager.m_pBuffIdVec, -1, sizeof(int) * manager.m_maxNumofSplittable);
	manager.Memset(manager.m_pNumofBuffId, 0, sizeof(int));
	InitNodeStat<<<1, 1>>>(nNumofIns, manager.m_pGrad, manager.m_pHess,
						   manager.m_pSNodeStat, manager.m_pSNIdToBuffId, manager.m_maxNumofSplittable,
						   manager.m_pBuffIdVec, manager.m_pNumofBuffId);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in InitNodeStat" << endl;
		exit(0);
	}
#endif
}

