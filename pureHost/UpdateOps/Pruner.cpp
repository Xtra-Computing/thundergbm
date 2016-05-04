/*
 * Prune.cpp
 *
 *  Created on: 2 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "Pruner.h"
#include "../MyAssert.h"

using std::cout;
using std::endl;

double Pruner::min_loss = -1;

/**
 * @brief: prune leaf that has gain smaller than the minimum gain (i.e. loss)
 */
void Pruner::pruneLeaf(RegTree &tree)
{
	leafChildCnt.clear();
	markDelete.clear();
	int nNumofNode = tree.nodes.size();
	leafChildCnt.resize(nNumofNode);
	markDelete.resize(nNumofNode);

	int npruned = 0;
    for(int nid = 0; nid < nNumofNode; ++nid)
    {
      if(tree.nodes[nid]->isLeaf())
      {
        npruned = this->TryPruneLeaf(tree, nid, npruned);
      }
    }

    cout << npruned << " nodes are pruned" << endl;
}


int Pruner::TryPruneLeaf(RegTree &tree, int nid, int npruned)
{ // NOLINT(*)
	PROCESS_ERROR(nid >= 0 && tree.nodes.size() > 0);
	TreeNode* &node = tree.nodes[nid];

    if(node->isRoot())//root node cannot be pruned
    	return npruned;

    int pid = node->parentId;
    PROCESS_ERROR(pid > 0);
    leafChildCnt[pid]++;
    PROCESS_ERROR(min_loss > 0);

    //try pruning two leaf nodes
    if(leafChildCnt[pid] >= 2 && tree.nodes[pid]->loss < min_loss)
    {
    	PROCESS_ERROR(leafChildCnt[pid] == 2);
    	// need to be pruned
    	ChangeToLeaf(tree, pid, tree.nodes[pid]->base_weight);
    	// tail recursion
    	return this->TryPruneLeaf(tree, pid, npruned+2);
    }
    else
    {
      return npruned;
    }
}

/**
 * @brief: change an internal node to a leaf node
 */
void Pruner::ChangeToLeaf(RegTree &tree, int nid, double value)
{
	TreeNode* &node = tree.nodes[nid];
	markDelete[node->leftChildId] = -1;
	markDelete[node->rightChildId] = -1;

	node->featureId = -1;
	node->leftChildId = -1;
	node->rightChildId = -1;

	node->predValue = value;
}


