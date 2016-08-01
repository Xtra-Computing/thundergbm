/*
 * DeviceSplitterSplitNode.cu
 *
 *  Created on: 12 May 2016
 *      Author: Zeyi Wen
 *		@brief: GPU version of splitAll function
 */

#include <iostream>
#include <algorithm>

#include "../../DeviceHost/MyAssert.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../Memory/SNMemManager.h"
#include "DeviceSplitter.h"
#include "../Preparator.h"
#include "../Hashing.h"
#include "DeviceSplitAllKernel.h"
#include "../KernelConf.h"

using std::cout;
using std::cerr;
using std::endl;
using std::pair;
using std::make_pair;
using std::sort;


/**
 * @brief: split all splittable nodes of the current level
 * @numofNode: for computing new children ids
 */
void DeviceSplitter::SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
		 	 	 	    const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel)
{

	int preMaxNodeId = m_nNumofNode - 1;
	PROCESS_ERROR(preMaxNodeId >= 0);

	GBDTGPUMemManager manager;
	SNGPUManager snManager;//splittable node memory manager

	//compute the base_weight of tree node, also determines if a node is a leaf.
//	cout << "compute weight" << endl;
	KernelConf conf;
	int threadPerBlock;
	dim3 dimNumofBlock;
	conf.ConfKernel(manager.m_curNumofSplitable, threadPerBlock, dimNumofBlock);
	clock_t com_weight_start = clock();
	ComputeWeight<<<dimNumofBlock, threadPerBlock>>>(snManager.m_pTreeNode, manager.m_pSplittableNode, manager.m_pSNIdToBuffId,
			  	  	  	  	  manager.m_pBestSplitPoint, manager.m_pSNodeStat, rt_eps, LEAFNODE,
			  	  	  	  	  m_lambda, manager.m_curNumofSplitable, bLastLevel, manager.m_maxNumofSplittable);
	clock_t com_weight_end = clock();
	cudaDeviceSynchronize();
	total_weight_t += (com_weight_end - com_weight_start);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in ComputeWeight" << endl;
		exit(0);
	}
#endif
	if(bLastLevel == true)//don't need to do the rest?
		return;

//	cout << "create new nodes" << endl;
	//copy the number of nodes in the tree to the GPU memory
	manager.Memset(snManager.m_pNumofNewNode, 0, sizeof(int));
	clock_t new_node_start = clock();
	CreateNewNode<<<dimNumofBlock, threadPerBlock>>>(
							snManager.m_pTreeNode, manager.m_pSplittableNode, snManager.m_pNewSplittableNode,
							manager.m_pSNIdToBuffId, manager.m_pBestSplitPoint,
							snManager.m_pParentId, snManager.m_pLeftChildId, snManager.m_pRightChildId,
							manager.m_pLChildStat, manager.m_pRChildStat, snManager.m_pNewNodeStat,
							snManager.m_pCurNumofNode_d, snManager.m_pNumofNewNode, rt_eps,
							manager.m_curNumofSplitable, bLastLevel, manager.m_maxNumofSplittable);
	int newNode = -1;
	manager.MemcpyDeviceToHost(snManager.m_pNumofNewNode, &newNode, sizeof(int));
	clock_t new_node_end = clock();
	total_create_node_t += (new_node_end - new_node_start);
	if(newNode == 0)//not new nodes are constructed
		return;

#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in CreateNewNode" << endl;
		exit(0);
	}
#endif

//	cout << "get unique fid" << endl;
//	cout << "lock value is: " << lockValue << endl;
	//find all used unique feature ids. We will use these features to organise instances into new nodes.
	manager.Memset(snManager.m_pFeaIdToBuffId, -1, sizeof(int) * snManager.m_maxNumofUsedFea);
	manager.Memset(snManager.m_pUniqueFeaIdVec, -1, sizeof(int) * snManager.m_maxNumofUsedFea);
	manager.Memset(snManager.m_pNumofUniqueFeaId, 0, sizeof(int));
	if(dimNumofBlock.x > 1 || dimNumofBlock.y > 1 || dimNumofBlock.z > 1)
	{
		cerr << "Bug: block for get uniqueFid is too large " << dimNumofBlock.x << endl;
		exit(0);
	}
	clock_t unique_id_start = clock();
	GetUniqueFid<<<threadPerBlock, 1>>>(snManager.m_pTreeNode, manager.m_pSplittableNode, manager.m_curNumofSplitable,
							 snManager.m_pFeaIdToBuffId, snManager.m_pUniqueFeaIdVec, snManager.m_pNumofUniqueFeaId,
			 	 	 	 	 snManager.m_maxNumofUsedFea, LEAFNODE, manager.m_nSNLock);
	cudaDeviceSynchronize();
	clock_t unique_id_end = clock();
	total_unique_id_t += (unique_id_end - unique_id_start);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in GetUniqueFid" << endl;
		exit(0);
	}

	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error before InsToNewNode" << endl;
		exit(0);
	}
#endif

	//for each used feature to move instances to new nodes
	int numofUniqueFea = -1;
	manager.MemcpyDeviceToHost(snManager.m_pNumofUniqueFeaId, &numofUniqueFea, sizeof(int));

	if(numofUniqueFea == 0)
		PROCESS_ERROR(bLastLevel == true);
	if(numofUniqueFea > 0)//need to move instances to new nodes if there are new nodes.
	{
//		cout << "ins to new nodes" << endl;
		dim3 dimGridThreadForEachUsedFea;
		int thdPerBlockIns2node = -1;
		conf.ConfKernel(m_maxNumValuePerFea, thdPerBlockIns2node, dimGridThreadForEachUsedFea);
		PROCESS_ERROR(dimGridThreadForEachUsedFea.z == 1);
		dimGridThreadForEachUsedFea.z = numofUniqueFea;//a decision feature is handled by a set of blocks
		clock_t ins2node_start = clock();
		InsToNewNode<<<dimGridThreadForEachUsedFea, thdPerBlockIns2node>>>(
								 snManager.m_pTreeNode, manager.m_pdDFeaValue, manager.m_pDInsId,
								 manager.m_pFeaStartPos, manager.m_pDNumofKeyValue,
								 manager.m_pInsIdToNodeId, manager.m_pSNIdToBuffId, manager.m_pBestSplitPoint,
								 snManager.m_pUniqueFeaIdVec, snManager.m_pNumofUniqueFeaId,
								 snManager.m_pParentId, snManager.m_pLeftChildId, snManager.m_pRightChildId,
								 preMaxNodeId, manager.m_numofFea, manager.m_numofIns, LEAFNODE);
		cudaDeviceSynchronize();
		clock_t ins2node_end = clock();
		total_ins2node_t += (ins2node_end - ins2node_start);
	}
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in InsToNewNode" << endl;
		exit(0);
	}
#endif

//	cout << "ins to new node by default" << endl;
	//for those instances of unknown feature values.
	int threadPerBlockEachIns;
	dim3 dimNumofBlockEachIns;
	conf.ConfKernel(manager.m_numofIns, threadPerBlockEachIns, dimNumofBlockEachIns);

	clock_t ins2default_start = clock();
	InsToNewNodeByDefault<<<dimNumofBlockEachIns, threadPerBlockEachIns>>>(
									snManager.m_pTreeNode, manager.m_pInsIdToNodeId, manager.m_pSNIdToBuffId,
									snManager.m_pParentId, snManager.m_pLeftChildId,
			   	   	   	   	   	   	preMaxNodeId, manager.m_numofIns, LEAFNODE);
	cudaDeviceSynchronize();
	clock_t ins2default_end = clock();
	total_ins2default_t += (ins2default_end - ins2default_start);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in InsToNewNodeByDefault" << endl;
		exit(0);
	}
#endif

	//update new splittable nodes
	int numofNewSplittableNode = -1;
	manager.MemcpyDeviceToHost(snManager.m_pNumofNewNode, &numofNewSplittableNode, sizeof(int));
	if(numofNewSplittableNode == 0)
		PROCESS_ERROR(bLastLevel == true);
	if(numofNewSplittableNode > 0)//update splittable nodes when there are new splittable nodes
	{
		dim3 dimGridThreadForEachNewSN;
		conf.ComputeBlock(numofNewSplittableNode, dimGridThreadForEachNewSN);
		int sharedMemSizeNSN = 1;

//		cout << "update new splittable" << endl;
		//reset nodeId to bufferId
		manager.Memset(manager.m_pSNIdToBuffId, -1, sizeof(int) * manager.m_maxNumofSplittable);
		manager.Memset(manager.m_pNumofBuffId, 0, sizeof(int));
		//reset nodeStat
		manager.Memset(manager.m_pSNodeStat, 0, sizeof(nodeStat) * manager.m_maxNumofSplittable);
		clock_t update_new_sp_start = clock();
		UpdateNewSplittable<<<dimGridThreadForEachNewSN, sharedMemSizeNSN>>>(
									  snManager.m_pNewSplittableNode, snManager.m_pNewNodeStat, manager.m_pSNIdToBuffId,
									  manager.m_pSNodeStat, snManager.m_pNumofNewNode, manager.m_pBuffIdVec, manager.m_pNumofBuffId,
									  manager.m_maxNumofSplittable, manager.m_nSNLock);
		cudaDeviceSynchronize();
		clock_t update_new_sp_end = clock();
		total_update_new_splittable_t += (update_new_sp_end - update_new_sp_start);
#if testing
	if(cudaGetLastError() != cudaSuccess)
	{
		cout << "error in UpdateNewSplittable" << endl;
		exit(0);
	}
#endif

		manager.MemcpyDeviceToDevice(snManager.m_pNewSplittableNode, manager.m_pSplittableNode, sizeof(TreeNode) * manager.m_maxNumofSplittable);
	}
//	cout << "Done split all" << endl;
}

/**
 * @brief: compute the maximum number of values of the features
 */
void DeviceSplitter::ComputeMaxNumValuePerFea(int *pnEachFeaLen, int numFea)
{
	m_maxNumValuePerFea = 0;
	for(int a = 0; a < numFea; a++)
	{
		if(m_maxNumValuePerFea < pnEachFeaLen[a])
			m_maxNumValuePerFea = pnEachFeaLen[a];
	}
}
