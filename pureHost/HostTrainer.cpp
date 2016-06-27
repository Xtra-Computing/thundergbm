/*
 * Trainer.cpp
 *
 *  Created on: 6 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: GBDT trainer implementation
 */

#include "HostTrainer.h"

/**
 * @brief: grow the tree by splitting nodes to the full extend
 */
void HostTrainer::GrowTree(RegTree &tree)
{
	//start splitting this tree from the root node
	vector<TreeNode*> splittableNode;
	for(int i = 0; i < int(tree.nodes.size()); i++)
	{
		splittableNode.push_back(tree.nodes[i]);
	}

	//split node(s)
	int nCurDepth = 0;
	while(splittableNode.size() > 0 && nCurDepth <= m_nMaxDepth)
	{
		splitter->m_nCurDept = nCurDepth;
//		cout << "splitting " << nCurDepth << " level..." << endl;
		vector<SplitPoint> vBest;

		vector<nodeStat> rchildStat, lchildStat;
		int bufferSize = splitter->mapNodeIdToBufferPos.size();//maps node id to buffer position
		vBest.resize(bufferSize);
		rchildStat.resize(bufferSize);
		lchildStat.resize(bufferSize);

		//efficient way to find the best split
		clock_t begin_find_fea = clock();
		splitter->FeaFinderAllNode(vBest, rchildStat, lchildStat);

		clock_t end_find_fea = clock();
		total_find_fea_t += (double(end_find_fea - begin_find_fea) / CLOCKS_PER_SEC);

		//split all the splittable nodes
		clock_t start_split_t = clock();
		bool bLastLevel = false;
		if(nCurDepth == m_nMaxDepth)
			bLastLevel = true;
		splitter->SplitAll(splittableNode, vBest, tree, m_nNumofNode, rchildStat, lchildStat, bLastLevel);
		clock_t end_split_t = clock();
		total_split_t += (double(end_split_t - start_split_t) / CLOCKS_PER_SEC);

		nCurDepth++;
	}

	clock_t begin_prune = clock();
	int numofNode = tree.nodes.size();
	TreeNode ** temp = &tree.nodes[0];
	pruner.pruneLeaf(temp, numofNode);
	clock_t end_prune = clock();
	total_prune_t += (double(end_prune - begin_prune) / CLOCKS_PER_SEC);
}

/**
 * @brief: initialise tree
 */
void HostTrainer::InitTree(RegTree &tree)
{
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

	total_find_fea_t = 0;
	total_split_t = 0;
	total_prune_t = 0;
}

/**
 * @brief: release memory used by trees
 */
void HostTrainer::ReleaseTree(vector<RegTree> &v_Tree)
{
	int nNumofTree = v_Tree.size();
	for(int i = 0; i < nNumofTree; i++)
	{
		int nNumofNodes = v_Tree[i].nodes.size();
		for(int j = 0; j < nNumofNodes; j++)
		{
			delete[] v_Tree[i].nodes[j];
		}
	}
}
