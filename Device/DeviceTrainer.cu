/*
 * DeviceTrainer.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "DeviceTrainer.h"
#include "Splitter/Initiator.h"
#include "Splitter/DeviceSplitter.h"
#include "Memory/gbdtGPUMemManager.h"
#include "Bagging/BagManager.h"
#include "FindSplit/IndexComputer.h"
#include "CSR/CsrCompressor.h"
#include "../SharedUtility/CudaMacro.h"

/**
 * @brief: initialise tree
 */
void DeviceTrainer::InitTree(RegTree &tree, void *pStream, int bagId)
{
	total_find_fea_t = 0;
	total_split_t = 0;
	total_prune_t = 0;

	//#### initial root node in GPU has been moved to grow tree.

	//all instances belong to the root node
	BagManager bagManager;
	cudaMemsetAsync(bagManager.m_pInsIdToNodeIdEachBag + bagId * bagManager.m_numIns, 0, sizeof(int) * bagManager.m_numIns, (*(cudaStream_t*)pStream));
}

/**
 * @brief: release memory used by trees
 */
void DeviceTrainer::ReleaseTree(vector<RegTree> &v_Tree)
{
	int nNumofTree = v_Tree.size();
	for(int i = 0; i < nNumofTree; i++)
	{
		int nNumofNodes = v_Tree[i].nodes.size();
		delete[] v_Tree[i].nodes[0];
	}
}

/**
 * @brief: grow the tree by splitting nodes to the full extend
 */
void DeviceTrainer::GrowTree(RegTree &tree, void *pStream, int bagId)
{
	int nNumofSplittableNode = 0;

	clock_t init_start = clock();
	//copy the root node to GPU
	BagManager bagManager;
	GBDTGPUMemManager manager;
	InitRootNode<<<1, 1, 0, (*(cudaStream_t*)pStream)>>>(bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
									bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, bagManager.m_numIns);

	manager.MemcpyDeviceToDeviceAsync(bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
								  	  bagManager.m_pSplittableNodeEachBag + bagId * bagManager.m_maxNumSplittable,
								  	  sizeof(TreeNode), pStream);
	clock_t init_end = clock();
	total_init_t += (init_end - init_start);

	nNumofSplittableNode++;
	//manager.m_curNumofSplitable = 1;
	bagManager.m_curNumofSplitableEachBag_h[bagId] = 1;

	vector<TreeNode*> splittableNode;

	//split node(s)
	int nCurDepth = 0;
	DeviceSplitter *pDSpliter = (DeviceSplitter*)splitter;
#ifdef _DEBUG
	pDSpliter->total_scan_t = 0;
	pDSpliter->total_com_gain_t = 0;
	pDSpliter->total_fill_gd_t = 0;
	pDSpliter->total_search_t = 0;
	pDSpliter->total_fix_gain_t = 0;
	pDSpliter->total_com_idx_t = 0;
	pDSpliter->total_csr_len_t = 0;
	pDSpliter->total_weight_t = 0;
	pDSpliter->total_create_node_t = 0;
	pDSpliter->total_unique_id_t = 0;
	pDSpliter->total_ins2node_t = 0;
	pDSpliter->total_ins2default_t = 0;
	pDSpliter->total_update_new_splittable_t = 0;
#endif
	while(bagManager.m_curNumofSplitableEachBag_h[bagId] > 0 && nCurDepth <= m_nMaxDepth)
	{
		pDSpliter->m_nCurDept = nCurDepth;
//		cout << "splitting " << nCurDepth << " level..." << endl;

		vector<SplitPoint> vBest;
		vector<nodeStat> rchildStat, lchildStat;
		cudaStreamSynchronize((*(cudaStream_t*)pStream));
		clock_t begin_find_fea = clock();

		if(nCurDepth < m_nMaxDepth){//don't need to find split for the last level
			if(CsrCompressor::bUseCsr == true)
				pDSpliter->FeaFinderAllNode2(pStream, bagId);
			else
				pDSpliter->FeaFinderAllNode(pStream, bagId);
		}

		clock_t end_find_fea = clock();
		total_find_fea_t += (double(end_find_fea - begin_find_fea) / CLOCKS_PER_SEC);

		//split all the splittable nodes
		clock_t start_split_t = clock();
		bool bLastLevel = false;
		if(nCurDepth == m_nMaxDepth)
			bLastLevel = true;

		int curNumofNode = -1;//this is fine even though bagging is used, as each bag is handled by a host thread.
		manager.MemcpyDeviceToHostAsync(bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, &curNumofNode, sizeof(int), pStream);
		PROCESS_ERROR(curNumofNode > 0);
		pDSpliter->SplitAll(curNumofNode, bLastLevel, pStream, bagId);

		manager.MemcpyDeviceToHostAsync(bagManager.m_pNumofNewNodeTreeOnTrainingEachBag + bagId, bagManager.m_curNumofSplitableEachBag_h + bagId,
								   sizeof(int), pStream);
		clock_t end_split_t = clock();
		total_split_t += (double(end_split_t - start_split_t) / CLOCKS_PER_SEC);
		nCurDepth++;
	}

	//copy tree nodes back to host
	clock_t begin_prune = clock();
	int numofNode = 0;
	manager.MemcpyDeviceToHostAsync(bagManager.m_pCurNumofNodeTreeOnTrainingEachBag_d + bagId, &numofNode,
									sizeof(int), pStream);
	cout << "number of nodes " << numofNode << endl;
	TreeNode *pAllNode = new TreeNode[numofNode];
	manager.MemcpyDeviceToHostAsync(bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
									pAllNode, sizeof(TreeNode) * numofNode, pStream);
	TreeNode **ypAllNode = new TreeNode*[numofNode];
	PROCESS_ERROR(tree.nodes.size() == 0);
	int nDefault2right = 0;
	for(int n = 0; n < numofNode; n++){
		ypAllNode[n] = &pAllNode[n];
		tree.nodes.push_back(&pAllNode[n]);//for getting features of trees
		if(pAllNode[n].m_bDefault2Right == true)
			nDefault2right++;
	}
	printf("default to right %d\n", nDefault2right);
	pruner.pruneLeaf(ypAllNode, numofNode);
	delete []ypAllNode;
	//########### can be improved by storing only the valid nodes afterwards

	StoreFinalTree(pAllNode, numofNode, pStream, bagId);

#ifdef _DEBUG
	clock_t end_prune = clock();
	total_prune_t += (double(end_prune - begin_prune) / CLOCKS_PER_SEC);

	double total_scan = pDSpliter->total_scan_t;
	double total_gain = pDSpliter->total_com_gain_t;
	double total_fill = pDSpliter->total_fill_gd_t;
	double total_search = pDSpliter->total_search_t;
	double total_fix = pDSpliter->total_fix_gain_t;
	double total_com_idx = pDSpliter->total_com_idx_t;
	cout << "com idx " << total_com_idx/CLOCKS_PER_SEC
		 << "; com csr len " << pDSpliter->total_csr_len_t/CLOCKS_PER_SEC
		 << "; scan takes " << total_scan/CLOCKS_PER_SEC << "; comp gain takes " << total_gain/CLOCKS_PER_SEC
		 << "; fix gain takes " << total_fix / CLOCKS_PER_SEC
		 << "; fill gd takes " << total_fill/CLOCKS_PER_SEC << "; search takes " << total_search/CLOCKS_PER_SEC << endl;

	//split
	double total_weight = pDSpliter->total_weight_t;
	double total_create_node = pDSpliter->total_create_node_t;
	double total_unique_id = pDSpliter->total_unique_id_t;
	double total_ins2node = pDSpliter->total_ins2node_t;
	double total_ins2default = pDSpliter->total_ins2default_t;
	double total_update_new_sp = pDSpliter->total_update_new_splittable_t;
	cout << "comp weight " << total_weight/CLOCKS_PER_SEC
		 << "; create node " << total_create_node/CLOCKS_PER_SEC
		 << "; unique id " << total_unique_id/CLOCKS_PER_SEC
		 << "; ins2node " << total_ins2node/CLOCKS_PER_SEC
		 << "; ins2default " << total_ins2default/CLOCKS_PER_SEC
		 << "; update new splittable " << total_update_new_sp/CLOCKS_PER_SEC << endl;
#endif
}

/**
 * @brief: store the tree learned at this round to GPU memory
 */
void DeviceTrainer::StoreFinalTree(TreeNode *pAllNode, int numofNode, void *pStream, int bagId)
{
	BagManager bagManager;
	GBDTGPUMemManager manager;
	//copy the final tree to GPU memory
	manager.MemcpyHostToDeviceAsync(pAllNode, bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
									sizeof(TreeNode) * numofNode, pStream);

	//copy the final tree for ensembling
	int numofTreeLearnt = manager.m_pNumofTreeLearntEachBag_h[bagId];
	int curLearningTreeId = numofTreeLearnt;
	manager.MemcpyHostToDeviceAsync(&numofNode, manager.m_pNumofNodeEachTreeEachBag + bagId * bagManager.m_numTreeEachBag + curLearningTreeId,
									sizeof(int), pStream);
	int numofNodePreviousTree = 0;
	int previousTreeStartPosInBag = bagId * bagManager.m_numTreeEachBag * bagManager.m_maxNumNode;
	if(numofTreeLearnt > 0)
	{
		int lastLearntTreeId = numofTreeLearnt - 1;
		manager.MemcpyDeviceToHostAsync(manager.m_pNumofNodeEachTreeEachBag + bagId * bagManager.m_numTreeEachBag + lastLearntTreeId,
										&numofNodePreviousTree, sizeof(int), pStream);
		manager.MemcpyDeviceToHostAsync(manager.m_pStartPosOfEachTreeEachBag + bagId * bagManager.m_numTreeEachBag + lastLearntTreeId,
										&previousTreeStartPosInBag, sizeof(int), pStream);
	}
	int treeStartPos = previousTreeStartPosInBag + numofNodePreviousTree;
	manager.MemcpyHostToDeviceAsync(&treeStartPos, manager.m_pStartPosOfEachTreeEachBag + bagId * bagManager.m_numTreeEachBag + curLearningTreeId,
									sizeof(int), pStream);
	manager.MemcpyDeviceToDeviceAsync(bagManager.m_pNodeTreeOnTrainingEachBag + bagId * bagManager.m_maxNumNode,
										manager.m_pAllTreeEachBag + treeStartPos, sizeof(TreeNode) * numofNode, pStream);
	manager.m_pNumofTreeLearntEachBag_h[bagId]++;
}
