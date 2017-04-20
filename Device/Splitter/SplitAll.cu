/*
 * DeviceSplitterSplitNode.cu
 *
 *  Created on: 12 May 2016
 *      Author: Zeyi Wen
 *		@brief: GPU version of splitAll function
 */

#include <iostream>
#include <algorithm>

#include "DeviceSplitter.h"
#include "DeviceSplitAllKernel.h"
#include "../Preparator.h"
#include "../Hashing.h"
#include "../Bagging/BagManager.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../../DeviceHost/MyAssert.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/CudaMacro.h"

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
		 	 	 	    const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel, void *pStream, int bagId)
{
	int preMaxNodeId = m_nNumofNode - 1;
	PROCESS_ERROR(preMaxNodeId >= 0);
	BagManager bagManager;
	bagManager.m_pPreMaxNid_h[bagId] = preMaxNodeId;
	GBDTGPUMemManager manager;

	KernelConf conf;
	int threadPerBlock;
	dim3 dimNumofBlock;
	conf.ConfKernel(bagManager.m_curNumofSplitableEachBag_h[bagId], threadPerBlock, dimNumofBlock);
	clock_t com_weight_start = clock();
	ComputeWeight<<<dimNumofBlock, threadPerBlock, 0, (*(cudaStream_t*)pStream)>>>(
							  bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
							  bagManager.m_pSplittableNodeEachBag + bagId * bagManager.m_maxNumSplittable,
							  bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
						  	  bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
						  	  rt_eps, LEAFNODE, m_lambda,
						  	  bagManager.m_curNumofSplitableEachBag_h[bagId], bLastLevel, bagManager.m_maxNumSplittable);
	clock_t com_weight_end = clock();
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	total_weight_t += (com_weight_end - com_weight_start);
	GETERROR("after ComputeWeight");

	if(bLastLevel == true)//don't need to do the rest?
		return;

	//copy the number of nodes in the tree to the GPU memory
	manager.MemsetAsync(bagManager.m_pNumofNewNodeTreeOnTrainingEachBag + bagId, 0, sizeof(int), pStream);
	clock_t new_node_start = clock();
	CreateNewNode<<<dimNumofBlock, threadPerBlock, 0, (*(cudaStream_t*)pStream)>>>(
							bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
							bagManager.m_pSplittableNodeEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pNewSplittableNodeEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pParentIdEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pLeftChildIdEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pRightChildIdEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pLChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pRChildStatEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pNewNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
							bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, bagManager.m_pNumofNewNodeTreeOnTrainingEachBag + bagId, rt_eps,
							bagManager.m_curNumofSplitableEachBag_h[bagId], bLastLevel, bagManager.m_maxNumSplittable);
	int newNode = -1;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pNumofNewNodeTreeOnTrainingEachBag + bagId, &newNode, sizeof(int), pStream);
	clock_t new_node_end = clock();
	total_create_node_t += (new_node_end - new_node_start);
	if(newNode == 0)//not new nodes are constructed
		return;
	GETERROR("in CreateNewNode");

	//find all used unique feature ids. We will use these features to organise instances into new nodes.
	manager.MemsetAsync(bagManager.m_pFeaIdToBuffIdEachBag + bagId * bagManager.m_maxNumUsedFeaATree, -1, sizeof(int) * bagManager.m_maxNumUsedFeaATree, pStream);
	manager.MemsetAsync(bagManager.m_pUniqueFeaIdVecEachBag + bagId * bagManager.m_maxNumUsedFeaATree, -1, sizeof(int) * bagManager.m_maxNumUsedFeaATree, pStream);
	manager.MemsetAsync(bagManager.m_pNumofUniqueFeaIdEachBag + bagId, 0, sizeof(int), pStream);

	clock_t unique_id_start = clock();
	GetUniqueFid<<<1, bagManager.m_curNumofSplitableEachBag_h[bagId], 0, (*(cudaStream_t*)pStream)>>>(
							 bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
							 bagManager.m_pSplittableNodeEachBag + bagId * bagManager.m_maxNumSplittable,
							 bagManager.m_curNumofSplitableEachBag_h[bagId],
							 bagManager.m_pFeaIdToBuffIdEachBag + bagId * bagManager.m_maxNumUsedFeaATree,
							 bagManager.m_pUniqueFeaIdVecEachBag + bagId * bagManager.m_maxNumUsedFeaATree,
							 bagManager.m_pNumofUniqueFeaIdEachBag + bagId,
							 bagManager.m_maxNumUsedFeaATree, LEAFNODE, bagManager.m_nSNLockEachBag + bagId);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	clock_t unique_id_end = clock();
	total_unique_id_t += (unique_id_end - unique_id_start);
	GETERROR("in GetUniqueFid");

	//for each used feature to move instances to new nodes
	int numofUniqueFea = -1;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pNumofUniqueFeaIdEachBag + bagId, &numofUniqueFea, sizeof(int), pStream);

	if(numofUniqueFea == 0)
		PROCESS_ERROR(bLastLevel == true);

	if(numofUniqueFea > 0)//need to move instances to new nodes if there are new nodes.
	{
//		cout << "ins to new nodes" << endl;
		dim3 dimGridThreadForEachUsedFea;
		int thdPerBlockIns2node = -1;
		conf.ConfKernel(bagManager.m_numIns, thdPerBlockIns2node, dimGridThreadForEachUsedFea);//######## can be improved: bagManager.m_numIns is the upper limits.
		PROCESS_ERROR(dimGridThreadForEachUsedFea.z == 1);
		dimGridThreadForEachUsedFea.z = numofUniqueFea;//a decision feature is handled by a set of blocks

		clock_t ins2node_start = clock();
		InsToNewNode<<<dimGridThreadForEachUsedFea, thdPerBlockIns2node, 0, (*(cudaStream_t*)pStream)>>>(
								 bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode, manager.m_pdDFeaValue, manager.m_pDInsId,
								 manager.m_pFeaStartPos, manager.m_pDNumofKeyValue,
								 bagManager.m_pSNIdToBuffIdEachBag + bagId * bagManager.m_maxNumSplittable,
								 bagManager.m_pBestSplitPointEachBag + bagId * bagManager.m_maxNumSplittable,
								 bagManager.m_pUniqueFeaIdVecEachBag + bagId * bagManager.m_maxNumUsedFeaATree,
								 bagManager.m_pNumofUniqueFeaIdEachBag + bagId,
								 bagManager.m_pParentIdEachBag + bagId * bagManager.m_maxNumSplittable,
								 bagManager.m_pLeftChildIdEachBag + bagId * bagManager.m_maxNumSplittable,
								 bagManager.m_pRightChildIdEachBag + bagId * bagManager.m_maxNumSplittable,
								 preMaxNodeId, manager.m_numofFea,
								 bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns,
								 manager.m_numofIns, LEAFNODE);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t ins2node_end = clock();
		total_ins2node_t += (ins2node_end - ins2node_start);
	}
	GETERROR("in InsToNewNode");

	//for those instances of unknown feature values.
	int threadPerBlockEachIns;
	dim3 dimNumofBlockEachIns;
	conf.ConfKernel(manager.m_numofIns, threadPerBlockEachIns, dimNumofBlockEachIns);

	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	
	clock_t ins2default_start = clock();
	InsToNewNodeByDefault<<<dimNumofBlockEachIns, threadPerBlockEachIns, 0, (*(cudaStream_t*)pStream)>>>(
									bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
									bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns,
									bagManager.m_pSNIdToBuffIdEachBag + bagId * bagManager.m_maxNumSplittable,
									bagManager.m_pParentIdEachBag + bagId * bagManager.m_maxNumSplittable,
									bagManager.m_pLeftChildIdEachBag + bagId * bagManager.m_maxNumSplittable,
			   	   	   	   	   	   	preMaxNodeId, manager.m_numofIns, LEAFNODE);
	cudaStreamSynchronize((*(cudaStream_t*)pStream));
	clock_t ins2default_end = clock();
	total_ins2default_t += (ins2default_end - ins2default_start);
	cudaDeviceSynchronize();

	GETERROR("in InsToNewNodeByDefault");

	//update new splittable nodes
	int numofNewSplittableNode = -1;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pNumofNewNodeTreeOnTrainingEachBag + bagId, &numofNewSplittableNode, sizeof(int), pStream);
	if(numofNewSplittableNode == 0)
		PROCESS_ERROR(bLastLevel == true);
	if(numofNewSplittableNode > 0)//update splittable nodes when there are new splittable nodes
	{
		dim3 dimGridThreadForEachNewSN;
		conf.ComputeBlock(numofNewSplittableNode, dimGridThreadForEachNewSN);
		int blockSizeNSN = 1;

//		printf("new sn=%d, blocksize=%d, blocks=%d\n", numofNewSplittableNode, blockSizeNSN, dimGridThreadForEachNewSN.x * dimGridThreadForEachNewSN.y * dimGridThreadForEachNewSN.z);
		//reset nodeId to bufferId
		manager.MemsetAsync(bagManager.m_pSNIdToBuffIdEachBag + bagId * bagManager.m_maxNumSplittable, -1,
							sizeof(int) * bagManager.m_maxNumSplittable, pStream);
		manager.MemsetAsync(bagManager.m_pNumofBuffIdEachBag + bagId, 0, sizeof(int), pStream);
		//reset nodeStat
		manager.MemsetAsync(bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable, 0,
						sizeof(nodeStat) * bagManager.m_maxNumSplittable, pStream);
		clock_t update_new_sp_start = clock();
		UpdateNewSplittable<<<dimGridThreadForEachNewSN, blockSizeNSN, 0, (*(cudaStream_t*)pStream)>>>(
									  bagManager.m_pNewSplittableNodeEachBag + bagId * bagManager.m_maxNumSplittable,
									  bagManager.m_pNewNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
									  	  bagManager.m_pSNIdToBuffIdEachBag + bagId * bagManager.m_maxNumSplittable,
									  bagManager.m_pSNodeStatEachBag + bagId * bagManager.m_maxNumSplittable,
									  	  bagManager.m_pNumofNewNodeTreeOnTrainingEachBag + bagId,
									  	  bagManager.m_pPartitionId2SNPosEachBag + bagId * bagManager.m_maxNumSplittable, bagManager.m_pNumofBuffIdEachBag + bagId,
									  bagManager.m_maxNumSplittable, bagManager.m_nSNLockEachBag + bagId,
									  preMaxNodeId);
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t update_new_sp_end = clock();
		total_update_new_splittable_t += (update_new_sp_end - update_new_sp_start);
		GETERROR("in UpdateNewSplittable");

		manager.MemcpyDeviceToDeviceAsync(bagManager.m_pNewSplittableNodeEachBag + bagId * bagManager.m_maxNumSplittable,
									 bagManager.m_pSplittableNodeEachBag + bagId * bagManager.m_maxNumSplittable,
									 sizeof(TreeNode) * bagManager.m_maxNumSplittable, pStream);
	}
//	cout << "Done split all" << endl;
}
