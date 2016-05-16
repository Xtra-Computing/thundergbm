/*
 * SplitAllKernel.cu
 *
 *  Created on: 15 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "DeviceSplitAllKernel.h"
#include "../Memory/gbdtGPUMemManager.h"

__global__ void ComputeWeight(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, int *pSNIdToBufferId,
								  SplitPoint *pBestSplitPoint, nodeStat *pSNodeStat, float_point rt_eps, int flag_LEAFNODE,
								  float_point lambda, int numofSplittableNode, bool bLastLevel)
{
	for(int n = 0; n < numofSplittableNode; n++)
	{
		int nid = pSplittableNode[n].nodeId;
		if(nid < 0)
		{
			printf("Error in %s: nid=%d\n", __PRETTY_FUNCTION__, nid);
			return;
		}
//		cout << "node " << nid << " needs to split..." << endl;
		int bufferPos = pSNIdToBufferId[nid];
		if(bufferPos < 0 || bufferPos >= numofSplittableNode)
		{
			printf("Error in %s: bufferPos=%d\n", __PRETTY_FUNCTION__, bufferPos);
			return;
		}

		//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
		pAllTreeNode[nid].loss = pBestSplitPoint[bufferPos].m_fGain;

		if(pSNodeStat[bufferPos].sum_hess <= 0)
		{
			printf("Error in %s: pSNodeStat[bufferPos].sum_hess=%d\n", __PRETTY_FUNCTION__, pSNodeStat[bufferPos].sum_hess);
			return;
		}

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
