/*
 * SplitAllKernel.cu
 *
 *  Created on: 15 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <string.h>
#include "DeviceSplitAllKernel.h"
#include "../Memory/gbdtGPUMemManager.h"

using std::string;

__device__ void ErrorChecker(int value, const char* functionName, const char* temp)
{
	if(value < 0)
	{
		printf("Error in %s: %s=%d\n", functionName, temp, value);
	}
}

/**
 * @brief: compute the base_weight of tree node, also determines if a node is a leaf.
 */
__global__ void ComputeWeight(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, int *pSNIdToBufferId,
								  SplitPoint *pBestSplitPoint, nodeStat *pSNodeStat, float_point rt_eps, int flag_LEAFNODE,
								  float_point lambda, int numofSplittableNode, bool bLastLevel)
{
	for(int n = 0; n < numofSplittableNode; n++)
	{
		int nid = pSplittableNode[n].nodeId;
		ErrorChecker(nid, __PRETTY_FUNCTION__, "nid");

//		cout << "node " << nid << " needs to split..." << endl;
		int bufferPos = pSNIdToBufferId[nid];
		ErrorChecker(bufferPos, __PRETTY_FUNCTION__, "bufferPos");

		//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
		pAllTreeNode[nid].loss = pBestSplitPoint[bufferPos].m_fGain;
		ErrorChecker(pSNodeStat[bufferPos].sum_hess, __PRETTY_FUNCTION__, "pSNodeStat[bufferPos].sum_hess");

		float_point nodeWeight = (-pSNodeStat[bufferPos].sum_gd / (pSNodeStat[bufferPos].sum_hess + lambda));
		pAllTreeNode[nid].base_weight = nodeWeight;
		if(pBestSplitPoint[bufferPos].m_fGain <= rt_eps || bLastLevel == true)
		{
			//weight of a leaf node
			pAllTreeNode[nid].predValue = pAllTreeNode[nid].base_weight;
			pAllTreeNode[nid].rightChildId = flag_LEAFNODE;
		}
	}

}

/**
 * @brief: create new nodes and associate new nodes with their parent id
 */
__global__ void CreateNewNode(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, TreeNode *pNewSplittableNode,
								  int *pSNIdToBufferId, SplitPoint *pBestSplitPoint,
								  int *pParentId, int *pLChildId, int *pRChildId,
								  nodeStat *pLChildStat, nodeStat *pRChildStat, nodeStat *pNewNodeStat,
								  int *m_nNumofNode,
								  float_point rt_eps, int nNumofSplittableNode, bool bLastLevel)
{
	//for each splittable node, assign lchild and rchild ids
//	vector<TreeNode*> newSplittableNode;

	int numofNewNode = 0;
	for(int n = 0; n < nNumofSplittableNode; n++)
	{
		int nid = pSplittableNode[n].nodeId;
		ErrorChecker(nid, __PRETTY_FUNCTION__, "nid");
		int bufferPos = pSNIdToBufferId[nid];
		ErrorChecker(bufferPos, __PRETTY_FUNCTION__, "bufferPos");

		if(!(pBestSplitPoint[bufferPos].m_fGain <= rt_eps || bLastLevel == true))
		{
			int lchildId = *m_nNumofNode;
			int rchildId = *m_nNumofNode + 1;

			//parent id to child ids
			pParentId[bufferPos] = nid;
			pLChildId[bufferPos] = lchildId;
			pRChildId[bufferPos] = rchildId;
			ErrorChecker(pLChildStat[bufferPos].sum_hess, __PRETTY_FUNCTION__, "lchildStat[bufferPos].sum_hess");
			ErrorChecker(pRChildStat[bufferPos].sum_hess, __PRETTY_FUNCTION__, "rchildStat[bufferPos].sum_hess");

			//push left and right child statistics into a vector
			int leftNewNodeId = numofNewNode;
			int rightNewNodeId = numofNewNode + 1;
			pNewNodeStat[leftNewNodeId] = pLChildStat[bufferPos];
			pNewNodeStat[rightNewNodeId] = pRChildStat[bufferPos];
			numofNewNode += 2;

			//split into two nodes
			TreeNode &leftChild = pAllTreeNode[lchildId];
			TreeNode &rightChild = pAllTreeNode[rchildId];
			int nLevel = pAllTreeNode[nid].level;

			leftChild.nodeId = lchildId;
			leftChild.parentId = nid;
			leftChild.level = nLevel + 1;
			rightChild.nodeId = rchildId;
			rightChild.parentId = nid;
			rightChild.level = nLevel + 1;

			//they should just be pointers, not new content
			pNewSplittableNode[leftNewNodeId] = leftChild;
			pNewSplittableNode[rightNewNodeId] = rightChild;


			pAllTreeNode[nid].leftChildId = leftChild.nodeId;
			pAllTreeNode[nid].rightChildId = rightChild.nodeId;
			ErrorChecker(pBestSplitPoint[bufferPos].m_nFeatureId + 1, __PRETTY_FUNCTION__, "pBestSplitPoint[bufferPos].m_nFeatureId");

			pAllTreeNode[nid].featureId = pBestSplitPoint[bufferPos].m_nFeatureId;
			pAllTreeNode[nid].fSplitValue = pBestSplitPoint[bufferPos].m_fSplitValue;

			m_nNumofNode += 2;
		}
	}

}
