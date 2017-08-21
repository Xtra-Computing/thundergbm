/*
 * Prune.cpp
 *
 *  Created on: 2 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>

#include "Pruner.h"
#include "../../SharedUtility/CudaMacro.h"

using std::cout;
using std::endl;

double Pruner::min_loss = -1;

/**
 * @brief: prune leaf that has gain smaller than the minimum gain (i.e. loss)
 */
void Pruner::pruneLeaf(TreeNode** nodes, int nNumofNode)
{
	leafChildCnt.clear();
	markDelete.clear();
	leafChildCnt.resize(nNumofNode);
	markDelete.resize(nNumofNode);

	int npruned = 0;
    for(int nid = 0; nid < nNumofNode; ++nid)
    {
      if(nodes[nid]->isLeaf())
      {
        npruned = this->TryPruneLeaf(nodes, nid, npruned);
      }
    }

    cout << npruned << " nodes are pruned" << endl;
}


int Pruner::TryPruneLeaf(TreeNode** nodes, int nid, int npruned)
{ // NOLINT(*)
	PROCESS_ERROR(nid >= 0 && nodes != NULL);
	TreeNode* &node = nodes[nid];

    if(node->isRoot())//root node cannot be pruned
    	return npruned;

    int parentId = node->parentId;
    PROCESS_ERROR(parentId >= 0);
    leafChildCnt[parentId]++;
    PROCESS_ERROR(min_loss > 0);

    //try pruning two leaf nodes
    if(leafChildCnt[parentId] >= 2 && nodes[parentId]->loss < min_loss)
    {
    	PROCESS_ERROR(leafChildCnt[parentId] == 2);
    	// need to be pruned
    	nodes[nodes[parentId]->leftChildId]->loss = -10.0;//mark the left child node as pruned
    	nodes[nodes[parentId]->rightChildId]->loss = -10.0;//mark the right child node as pruned
    	ChangeToLeaf(nodes[parentId], nodes[parentId]->base_weight);
    	// tail recursion
    	return this->TryPruneLeaf(nodes, parentId, npruned+2);
    }
    else
    {
      return npruned;
    }
}

/**
 * @brief: change an internal node to a leaf node
 */
void Pruner::ChangeToLeaf(TreeNode* node, double value)
{
//	TreeNode* &node = tree.nodes[nid];
	markDelete[node->leftChildId] = -1;
	markDelete[node->rightChildId] = -1;

//	cout << "node id " << node->leftChildId << " and " << node->rightChildId << " are pruned; new leaf is " << node->nodeId << endl;

	node->featureId = -1;
	node->leftChildId = -1;
	node->rightChildId = -1;

	node->predValue = value;
}


