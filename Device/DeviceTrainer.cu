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
#include "Memory/SplitNodeMemManager.h"
#include "Memory/dtMemManager.h"

/**
 * @brief: initialise tree
 */
void DeviceTrainer::InitTree(RegTree &tree)
{
	#ifdef _COMPARE_HOST
	TreeNode *root = new TreeNode[1];
	m_nNumofNode = 1;
	root->nodeId = 0;
	root->level = 0;

	tree.nodes.push_back(root);

	//all instances are under node 0
	splitter->m_nodeIds.clear();
	for(int i = 0; i < m_vvInsSparse.size(); i++)
	{
		splitter->m_nodeIds.push_back(0);
	}
	#endif

	total_find_fea_t = 0;
	total_split_t = 0;
	total_prune_t = 0;

	//#### initial root node in GPU has been moved to grow tree.

	//all instances belong to the root node
	GBDTGPUMemManager manager;
	cudaMemset(manager.m_pInsIdToNodeId, 0, sizeof(int) * manager.m_numofIns);
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
void DeviceTrainer::GrowTree(RegTree &tree)
{
	int nNumofSplittableNode = 0;

	//copy the root node to GPU
	GBDTGPUMemManager manager;
	SNGPUManager snManager;
	snManager.resetForNextTree();//reset tree nodes to default value

	InitRootNode<<<1, 1>>>(snManager.m_pTreeNode, snManager.m_pCurNumofNode);

	manager.MemcpyDeviceToDevice(snManager.m_pTreeNode, manager.m_pSplittableNode, sizeof(TreeNode));

	nNumofSplittableNode++;
	manager.m_curNumofSplitable = 1;

	vector<TreeNode*> splittableNode;

	//split node(s)
	int nCurDepth = 0;
	while(manager.m_curNumofSplitable > 0 && nCurDepth <= m_nMaxDepth)
	{
		splitter->m_nCurDept = nCurDepth;
//		cout << "splitting " << nCurDepth << " level..." << endl;

		vector<SplitPoint> vBest;
		vector<nodeStat> rchildStat, lchildStat;
		clock_t begin_find_fea = clock();

		splitter->FeaFinderAllNode(vBest, rchildStat, lchildStat);

		clock_t end_find_fea = clock();
		total_find_fea_t += (double(end_find_fea - begin_find_fea) / CLOCKS_PER_SEC);

		//split all the splittable nodes
		clock_t start_split_t = clock();
		bool bLastLevel = false;
		if(nCurDepth == m_nMaxDepth)
			bLastLevel = true;

		int curNumofNode = -1;
		manager.MemcpyDeviceToHost(snManager.m_pCurNumofNode, &curNumofNode, sizeof(int));
		PROCESS_ERROR(curNumofNode > 0);
		splitter->SplitAll(splittableNode, vBest, tree, curNumofNode, rchildStat, lchildStat, bLastLevel);
		clock_t end_split_t = clock();
		total_split_t += (double(end_split_t - start_split_t) / CLOCKS_PER_SEC);

		manager.MemcpyDeviceToHost(snManager.m_pNumofNewNode, &manager.m_curNumofSplitable, sizeof(int));
//		cout << "number of new/splittable nodes is " << manager.m_curNumofSplitable << endl;

		nCurDepth++;
	}

	//copy tree nodes back to host
	clock_t begin_prune = clock();
	int numofNode = 0;
	manager.MemcpyDeviceToHost(snManager.m_pCurNumofNode, &numofNode, sizeof(int));
	cout << "number of nodes " << numofNode << endl;
	TreeNode *pAllNode = new TreeNode[numofNode];
	manager.MemcpyDeviceToHost(snManager.m_pTreeNode, pAllNode, sizeof(TreeNode) * numofNode);
	TreeNode **ypAllNode = new TreeNode*[numofNode];
	for(int n = 0; n < numofNode; n++)
	{
		ypAllNode[n] = &pAllNode[n];
		tree.nodes.push_back(&pAllNode[n]);//for getting features of trees
	}
	pruner.pruneLeaf(ypAllNode, numofNode);
	delete []ypAllNode;
	//########### can be improved by storing only the valid nodes afterwards

	StoreFinalTree(pAllNode, numofNode);

	clock_t end_prune = clock();
	total_prune_t += (double(end_prune - begin_prune) / CLOCKS_PER_SEC);
}

/**
 * @brief: store the tree learned at this round to GPU memory
 */
void DeviceTrainer::StoreFinalTree(TreeNode *pAllNode, int numofNode)
{
	GBDTGPUMemManager manager;
	SNGPUManager snManager;
	//copy the final tree to GPU memory
	manager.MemcpyHostToDevice(pAllNode, snManager.m_pTreeNode, sizeof(TreeNode) * numofNode);

	//copy the final tree for ensembling
	DTGPUMemManager treeManager;
	int numofTreeLearnt = treeManager.m_numofTreeLearnt;
	int curLearningTreeId = numofTreeLearnt;
	manager.MemcpyHostToDevice(&numofNode, treeManager.m_pNumofNodeEachTree + curLearningTreeId, sizeof(int));
	int numofNodePreviousTree = 0;
	int previousStartPos = 0;
	if(numofTreeLearnt > 0)
	{
		int lastLearntTreeId = numofTreeLearnt - 1;
		manager.MemcpyDeviceToHost(treeManager.m_pNumofNodeEachTree + lastLearntTreeId, &numofNodePreviousTree, sizeof(int));
		manager.MemcpyDeviceToHost(treeManager.m_pStartPosOfEachTree + lastLearntTreeId, &previousStartPos, sizeof(int));
	}
	int treeStartPos = previousStartPos + numofNodePreviousTree;
	manager.MemcpyHostToDevice(&treeStartPos, treeManager.m_pStartPosOfEachTree + curLearningTreeId, sizeof(int));
	manager.MemcpyDeviceToDevice(snManager.m_pTreeNode, treeManager.m_pAllTree + treeStartPos, sizeof(TreeNode) * numofNode);
	treeManager.m_numofTreeLearnt++;
}
