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
		splitter.m_nCurDept = nCurDepth;
//		cout << "splitting " << nCurDepth << " level..." << endl;
		vector<SplitPoint> vBest;

		vector<nodeStat> rchildStat, lchildStat;
		int bufferSize = splitter.mapNodeIdToBufferPos.size();//maps node id to buffer position
		vBest.resize(bufferSize);
		rchildStat.resize(bufferSize);
		lchildStat.resize(bufferSize);

		//efficient way to find the best split
		clock_t begin_find_fea = clock();
		splitter.FeaFinderAllNode(vBest, rchildStat, lchildStat);

		clock_t end_find_fea = clock();
		total_find_fea_t += (double(end_find_fea - begin_find_fea) / CLOCKS_PER_SEC);

		//split all the splittable nodes
		clock_t start_split_t = clock();
		bool bLastLevel = false;
		if(nCurDepth == m_nMaxDepth)
			bLastLevel = true;
		splitter.SplitAll(splittableNode, vBest, tree, m_nNumofNode, rchildStat, lchildStat, bLastLevel);
		clock_t end_split_t = clock();
		total_split_t += (double(end_split_t - start_split_t) / CLOCKS_PER_SEC);

		nCurDepth++;
	}

	clock_t begin_prune = clock();
	pruner.pruneLeaf(tree);
	clock_t end_prune = clock();
	total_prune_t += (double(end_prune - begin_prune) / CLOCKS_PER_SEC);
}
