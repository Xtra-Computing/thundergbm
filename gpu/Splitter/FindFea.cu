/*
 * DeviceSplitter.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "../../pureHost/MyAssert.h"
#include "../../pureHost/SparsePred/DenseInstance.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../Memory/SplitNodeMemManager.h"
#include "DeviceSplitter.h"
#include "DeviceFindFeaKernel.h"
#include "../Preparator.h"
#include "DevicePredKernel.h"
#include "../Hashing.h"
#include "../DevicePrediction.h"

using std::cout;
using std::endl;
using std::make_pair;


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
	preparator.VecToArray(m_nodeIds, manager.m_pInsToNodeIdHost);
	manager.MemcpyHostToDevice(manager.m_pInsToNodeIdHost, manager.m_pInsIdToNodeId, sizeof(int) * manager.m_numofIns);

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
//		manager.MemcpyHostToDevice(&startPosOfCurFea, manager.m_pFeaStartPos + f, sizeof(long long));

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

/**
 * @brief: prediction and compute gradient descent
 */
void DeviceSplitter::ComputeGD(vector<RegTree> &vTree)
{
	GBDTGPUMemManager manager;
	SNGPUManager snManager;

	vector<double> v_fPredValue;

//	pred.PredictSparseIns(m_vvInsSparse, vTree, v_fPredValue, m_vPredBuffer);

	//get features and store the feature ids in a way that the access is efficient
	DenseInsConverter denseInsConverter(vTree);

	//hash feature id to position id
	int numofUsedFea = denseInsConverter.usedFeaSet.size();
	int *pHashUsedFea = NULL;
	int *pSortedUsedFea = NULL;
	if(numofUsedFea > 0)
	{
		pHashUsedFea = new int[numofUsedFea];
		memset(pHashUsedFea, -1, sizeof(int) * numofUsedFea);
		for(int uf = 0; uf < numofUsedFea; uf++)
		{
			bool bIsNewHashValue = false;
			int hashValue = Hashing::HostAssignHashValue(pHashUsedFea, denseInsConverter.usedFeaSet[uf], numofUsedFea, bIsNewHashValue);
			#ifdef _DEBUG
//			cout << "hash value of " << denseInsConverter.usedFeaSet[uf] << " is " << hashValue << endl;
			#endif
		}

		pSortedUsedFea = new int[numofUsedFea];
		for(int uf = 0; uf < numofUsedFea; uf++)
			pSortedUsedFea[uf] = denseInsConverter.usedFeaSet[uf];

		//copy hash map to gpu memory
		manager.MemcpyHostToDevice(pHashUsedFea, manager.m_pHashFeaIdToDenseInsPos, sizeof(int) * numofUsedFea);
		manager.MemcpyHostToDevice(pSortedUsedFea, manager.m_pSortedUsedFeaId, sizeof(int) * numofUsedFea);
	}

	//for each tree
	int nNumofTree = vTree.size();
	int nNumofIns = manager.m_numofIns;
	PROCESS_ERROR(nNumofIns > 0);

	//copy tree from GPU memory
	if(nNumofTree - 1 >= 0)
	{
		int numofNode = 0;
		manager.MemcpyDeviceToHost(snManager.m_pCurNumofNode, &numofNode, sizeof(int));
		TreeNode *pAllNode = new TreeNode[numofNode];
		manager.MemcpyDeviceToHost(snManager.m_pTreeNode, pAllNode, sizeof(TreeNode) * numofNode);

		#ifdef _DEBUG
		cout << numofNode << " v.s. " << vTree[nNumofTree - 1].nodes.size() << endl;
		//compare each node
		for(int n = 0; n < numofNode; n++)
		{
			if(!(pAllNode[n].nodeId == vTree[nNumofTree - 1].nodes[n]->nodeId
			   && pAllNode[n].featureId == vTree[nNumofTree - 1].nodes[n]->featureId
			   && pAllNode[n].fSplitValue == vTree[nNumofTree - 1].nodes[n]->fSplitValue))
			{
				cout << "node id: " << pAllNode[n].nodeId << " v.s. " << vTree[nNumofTree - 1].nodes[n]->nodeId
					 <<	"; feat id: " << pAllNode[n].featureId << " v.s. " << vTree[nNumofTree - 1].nodes[n]->featureId
					 << "; sp value: " << pAllNode[n].fSplitValue << " v.s. " << vTree[nNumofTree - 1].nodes[n]->fSplitValue
					 << "; rc id: " << pAllNode[n].rightChildId << " v.s. " << vTree[nNumofTree - 1].nodes[n]->rightChildId << endl;
			}
		}
		#endif
	}

	for(int i = 0; i < nNumofIns; i++)
	{
		double fValue = 0;
		checkCudaErrors(cudaMemset(manager.m_pTargetValue, 0, sizeof(float_point)));
		manager.MemcpyDeviceToHost(manager.m_pPredBuffer + i, &fValue, sizeof(float_point));

		//start prediction ###############

		vector<double> vDense;
		if(nNumofTree > 0)
		{
			long long startPos = -1;
			long long *pInsStartPos = manager.m_pInsStartPos + (long long)i;
			manager.MemcpyDeviceToHost(pInsStartPos, &startPos, sizeof(long long));
//			cout << "start pos ins" << i << "=" << startPos << endl;
			float_point *pDevInsValue = manager.m_pdDInsValue + startPos;
			int *pDevFeaId = manager.m_pDFeaId + startPos;
			int numofFeaValue = -1;
			int *pNumofFea = manager.m_pDNumofFea + i;
			manager.MemcpyDeviceToHost(pNumofFea, &numofFeaValue, sizeof(int));

			checkCudaErrors(cudaMemset(manager.m_pdDenseIns, 0, sizeof(float_point) * numofUsedFea));
			FillDense<<<1, 1>>>(pDevInsValue, pDevFeaId, numofFeaValue, manager.m_pdDenseIns,
								manager.m_pSortedUsedFeaId, manager.m_pHashFeaIdToDenseInsPos, numofUsedFea);

			//construct dense instance #### now for testing
			denseInsConverter.SparseToDense(vv_insDebug[i], vDense);
			//denseInsConverter.PrintDenseVec(vDense);

			//copy the dense instance to vector for testing
			float_point *pDense = new float_point[numofUsedFea];
			manager.MemcpyDeviceToHost(manager.m_pdDenseIns, pDense, sizeof(float_point) * numofUsedFea);

			bool bDiff = false;
			for(int i = 0; i < numofUsedFea; i++)
			{

				int pos = Hashing::HostGetBufferId(pHashUsedFea, pSortedUsedFea[i], numofUsedFea);
				if(vDense[i] != pDense[pos])
				{
					cout << "different: " << vDense[i] << " v.s. " << pDense[pos] << "\t";
					bDiff = true;
				}

				if(bDiff == true && (i == manager.m_numofFea - 1 || i == vDense.size() - 1))
					cout << endl;

				//vDense.push_back(pDense[i]);
			}
			////////end for testing
		}

		//prediction using the last tree
		if(nNumofTree - 1 >= 0)
		{
			int numofNode = 0;
			manager.MemcpyDeviceToHost(snManager.m_pCurNumofNode, &numofNode, sizeof(int));
			PredTarget<<<1, 1>>>(snManager.m_pTreeNode, numofNode, manager.m_pdDenseIns, numofUsedFea,
								 manager.m_pHashFeaIdToDenseInsPos, manager.m_pTargetValue);

			#ifdef _DEBUG
			//host prediction
			for(int t = nNumofTree - 1; t >= 0 && t < nNumofTree; t++)
			{
				int nodeId = vTree[t].GetLeafIdSparseInstance(vDense, denseInsConverter.fidToDensePos);
				fValue += vTree[t][nodeId]->predValue;
			}

			float_point fTarget = 0;
			manager.MemcpyDeviceToHost(manager.m_pTargetValue, &fTarget, sizeof(float_point));
			if(fValue != fTarget)
				cout << "Target value diff " << fValue << " v.s. " << fTarget << endl;
			#endif

		}

		v_fPredValue.push_back(fValue);
		manager.MemcpyDeviceToDevice(manager.m_pTargetValue, manager.m_pPredBuffer + i, sizeof(float_point));
	}

	if(pHashUsedFea == NULL)
		delete []pHashUsedFea;
	if(pSortedUsedFea == NULL)
		delete []pSortedUsedFea;

//	ComputeGDSparse(v_fPredValue, m_vTrueValue);
	//compute GD
	int nTotal = nNumofIns;
	for(int i = 0; i < nTotal; i++)
	{
		float_point fTrueValue = 0;
		manager.MemcpyDeviceToHost(manager.m_pdTrueTargetValue + i, &fTrueValue, sizeof(float_point));
		m_vGDPair_fixedPos[i].grad = v_fPredValue[i] - fTrueValue;
		m_vGDPair_fixedPos[i].hess = 1;
	}

	nodeStat rootStat;
	for(int i = 0; i < nTotal; i++)
	{
		rootStat.sum_gd += m_vGDPair_fixedPos[i].grad;
		rootStat.sum_hess += m_vGDPair_fixedPos[i].hess;
	}


	m_nodeStat.clear();
	m_nodeStat.push_back(rootStat);
	mapNodeIdToBufferPos.insert(make_pair(0,0));//node0 in pos0 of buffer
}

