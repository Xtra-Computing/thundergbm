/*
 * DeviceTrainer.cu
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "DeviceTrainer.h"
#include "Splitter/DeviceSplitter.h"
#include "Memory/gbdtGPUMemManager.h"

/**
 * @brief: grow the tree by splitting nodes to the full extend
 */
void DeviceTrainer::GrowTree(RegTree &tree)
{
	int nNumofSplittableNode = 0;

	//copy the root node to GPU
	GBDTGPUMemManager manager;
	manager.MemcpyHostToDevice(tree.nodes[0], manager.m_pSplittableNode, sizeof(TreeNode));
	nNumofSplittableNode++;

	vector<TreeNode*> splittableNode;
	for(int i = 0; i < int(tree.nodes.size()); i++)
	{
		splittableNode.push_back(tree.nodes[i]);
	}

	//split node(s)
	int nCurDepth = 0;
	while(nNumofSplittableNode > 0 && nCurDepth <= m_nMaxDepth)
	{
		splitter->m_nCurDept = nCurDepth;
//		cout << "splitting " << nCurDepth << " level..." << endl;

		int bufferSize = splitter->mapNodeIdToBufferPos.size();//maps node id to buffer position

		//efficient way to find the best split
		clock_t begin_find_fea = clock();
		vector<SplitPoint> vBest;
		vector<nodeStat> rchildStat, lchildStat;
		bufferSize = nNumofSplittableNode;//maps node id to buffer position
		vBest.resize(bufferSize);
		rchildStat.resize(bufferSize);
		lchildStat.resize(bufferSize);

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
		nNumofSplittableNode = splittableNode.size();

		nCurDepth++;
	}

	clock_t begin_prune = clock();
	pruner.pruneLeaf(tree);
	clock_t end_prune = clock();
	total_prune_t += (double(end_prune - begin_prune) / CLOCKS_PER_SEC);
}

