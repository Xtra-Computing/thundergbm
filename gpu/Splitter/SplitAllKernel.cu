/*
 * SplitAllKernel.cu
 *
 *  Created on: 15 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "DeviceSplitAllKernel.h"

__global__ void ComputeWeight(int nNumofSplittableNode)
{
/*	for(int n = 0; n < nNumofSplittableNode; n++)
	{
		int nid = splittableNode[n]->nodeId;
//		cout << "node " << nid << " needs to split..." << endl;
		int bufferPos = mapNodeIdToBufferPos[nid];
		PROCESS_ERROR(bufferPos < vBest.size());
		//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
		tree.nodes[nid]->loss = vBest[bufferPos].m_fGain;
		tree.nodes[nid]->base_weight = ComputeWeightSparseData(bufferPos);
		if(vBest[bufferPos].m_fGain <= rt_eps || bLastLevel == true)
		{
			//weight of a leaf node
			tree.nodes[nid]->predValue = tree.nodes[nid]->base_weight;
			tree.nodes[nid]->rightChildId = LEAFNODE;
		}
	}
	*/
}
