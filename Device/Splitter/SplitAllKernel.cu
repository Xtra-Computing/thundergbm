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
#include "../DeviceHashing.h"
#include "../ErrorChecker.h"

using std::string;

/**
 * @brief: compute the base_weight of tree node, also determines if a node is a leaf.
 */
__global__ void ComputeWeight(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, int *pSNIdToBufferId,
								  SplitPoint *pBestSplitPoint, nodeStat *pSNodeStat, float_point rt_eps, int flag_LEAFNODE,
								  float_point lambda, int numofSplittableNode, bool bLastLevel)
{
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId < 0 || nGlobalThreadId >= numofSplittableNode)//one thread per splittable node
		printf("Error in computeWeight function, thread id=%d\n", nGlobalThreadId);

	int nid = pSplittableNode[nGlobalThreadId].nodeId;
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

/**
 * @brief: create new nodes and associate new nodes with their parent id
 */
__global__ void CreateNewNode(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, TreeNode *pNewSplittableNode,
								 const int *pSNIdToBufferId, const SplitPoint *pBestSplitPoint,
								  int *pParentId, int *pLChildId, int *pRChildId,
								  const nodeStat *pLChildStat, const nodeStat *pRChildStat, nodeStat *pNewNodeStat,
								  int *pNumofNode, int *pNumofNewNode,
								  float_point rt_eps, int nNumofSplittableNode, bool bLastLevel)
{
	//for each splittable node, assign lchild and rchild ids
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId < 0 || nGlobalThreadId >= nNumofSplittableNode)//one thread per splittable node
		printf("Error in CreateNewNode function, thread id=%d\n", nGlobalThreadId);

	ErrorChecker(*pNumofNewNode == 0, __PRETTY_FUNCTION__, "*pNumofNewNode == 0");

	int nid = pSplittableNode[nGlobalThreadId].nodeId;
	ErrorChecker(nid, __PRETTY_FUNCTION__, "nid");
	int bufferPos = pSNIdToBufferId[nid];
	ErrorChecker(bufferPos, __PRETTY_FUNCTION__, "bufferPos");

	if(!(pBestSplitPoint[bufferPos].m_fGain <= rt_eps || bLastLevel == true))
	{
		int childrenId = atomicAdd(pNumofNode, 2);

		int lchildId = childrenId;
		int rchildId = childrenId + 1;

		//parent id to child ids
		pParentId[bufferPos] = nid;
		pLChildId[bufferPos] = lchildId;
		pRChildId[bufferPos] = rchildId;
		ErrorChecker(pLChildStat[bufferPos].sum_hess, __PRETTY_FUNCTION__, "lchildStat[bufferPos].sum_hess");
		ErrorChecker(pRChildStat[bufferPos].sum_hess, __PRETTY_FUNCTION__, "rchildStat[bufferPos].sum_hess");

		//push left and right child statistics into a vector
		int newNodeId = atomicAdd(pNumofNewNode, 2);
		int leftNewNodeId = newNodeId;
		int rightNewNodeId = newNodeId + 1;
		pNewNodeStat[leftNewNodeId] = pLChildStat[bufferPos];
		pNewNodeStat[rightNewNodeId] = pRChildStat[bufferPos];


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
		ErrorChecker(pBestSplitPoint[bufferPos].m_nFeatureId, __PRETTY_FUNCTION__, "pBestSplitPoint[bufferPos].m_nFeatureId");

		pAllTreeNode[nid].featureId = pBestSplitPoint[bufferPos].m_nFeatureId;
		pAllTreeNode[nid].fSplitValue = pBestSplitPoint[bufferPos].m_fSplitValue;

		//this is used in finding unique feature ids
		pSplittableNode[nGlobalThreadId].featureId = pBestSplitPoint[bufferPos].m_nFeatureId;

//			printf("cur # of node is %d\n", *pNumofNode);
	}

}

/**
 * @brief: get unique used feature ids of the splittable nodes
 */
__global__ void GetUniqueFid(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, int nNumofSplittableNode,
								 int *pFeaIdToBuffId, int *pUniqueFidVec, int *pNumofUniqueFid,
								 int maxNumofUsedFea, int flag_LEAFNODE, int *pnLock)
{
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId < 0 || nGlobalThreadId >= nNumofSplittableNode)//one thread per splittable node
		printf("Error in GetUniqueFid function, thread id=%d\n", nGlobalThreadId);

	ErrorCond(*pNumofUniqueFid == 0, __PRETTY_FUNCTION__, "*pNumofUniqueFid == 0");

	int fid = pSplittableNode[nGlobalThreadId].featureId;
	int nid = pSplittableNode[nGlobalThreadId].nodeId;
	if(fid == -1 && pAllTreeNode[nid].rightChildId == flag_LEAFNODE)
	{//leaf node should satisfy two conditions at this step
		return;
	}
	ErrorChecker(fid, __PRETTY_FUNCTION__, "fid");

	bool bLeaveLoop = false;
	while(bLeaveLoop == false)
	{
		//critical region when assigning hash value
		if(atomicExch(pnLock, 1) == 0)
		{
			bool bIsNew = false;
			int hashValue = AssignHashValue(pFeaIdToBuffId, fid, maxNumofUsedFea, bIsNew);
			if(bIsNew == true)
			{
				int numofUniqueFid = atomicAdd(pNumofUniqueFid, 1);
				pUniqueFidVec[numofUniqueFid] = fid;
			}
			ErrorChecker(hashValue, __PRETTY_FUNCTION__, "hashValue");
			bLeaveLoop = true;
			atomicExch(pnLock, 0);
		}
	}
}

/**
 * @brief: assign instances (which have non-zero values on the feature of interest) to new nodes
 */
__global__ void InsToNewNode(TreeNode *pAllTreeNode, float_point *pdFeaValue, int *pInsId,
								 long long *pFeaStartPos, int *pNumofKeyValue,
								 int *pInsIdToNodeId, int *pSNIdToBuffId, SplitPoint *pBestSplitPoint,
								 int *pUniqueFidVec, int *pNumofUniqueFid,
								 int *pParentId, int *pLChildId, int *pRChildId,
								 int preMaxNodeId, int numofFea, int numofIns, int flag_LEAFNODE)
{
	int numofUniqueFid = *pNumofUniqueFid;

	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId < 0 || nGlobalThreadId >= numofUniqueFid)//one thread per splittable node
		printf("Error in InsToNewNode function, thread id=%d\n", nGlobalThreadId);

	int ufid = pUniqueFidVec[nGlobalThreadId];
	ErrorChecker(ufid, __PRETTY_FUNCTION__, "ufid");
	ErrorChecker(numofFea - ufid, __PRETTY_FUNCTION__, "numofFea - ufid");

	//for each instance that has value on the feature
	long long curFeaStartPos = pFeaStartPos[ufid];
	float_point *pdCurFeaValue = pdFeaValue + curFeaStartPos;
	int *pCurFeaInsId = pInsId + curFeaStartPos;
	int nNumofPair = pNumofKeyValue[ufid];
	for(int i = 0; i < nNumofPair; i++)
	{
		int insId = pCurFeaInsId[i];
		ErrorChecker(insId, __PRETTY_FUNCTION__, "insId");
		ErrorChecker(numofIns - insId, __PRETTY_FUNCTION__, "numofIns - insId");
		int nid = pInsIdToNodeId[insId];

		if(nid < 0)//leaf node
			continue;

		if(nid > preMaxNodeId)//new node ids
			continue;

		ErrorChecker(nid, __PRETTY_FUNCTION__, "nid");
		int bufferPos = pSNIdToBuffId[nid];
		ErrorChecker(bufferPos, __PRETTY_FUNCTION__, "bufferPos");
		int fid = pBestSplitPoint[bufferPos].m_nFeatureId;
		if(fid != ufid)//this feature is not the splitting feature for the instance.
			continue;


		if(nid != pParentId[bufferPos])//node doesn't need to split (leaf node or new node)
		{
			if(pAllTreeNode[nid].rightChildId != flag_LEAFNODE)
			{
				ErrorChecker(preMaxNodeId - nid, __PRETTY_FUNCTION__, "preMaxNodeId - nid");
				continue;
			}
			ErrorCond(pAllTreeNode[nid].rightChildId == flag_LEAFNODE, __PRETTY_FUNCTION__, "pAllTreeNode[nid].rightChildId == flag_LEAFNODE");
			continue;
		}

		if(nid == pParentId[bufferPos])
		{//internal node (needs to split)
			ErrorCond(pRChildId[bufferPos] == pLChildId[bufferPos] + 1, __PRETTY_FUNCTION__, "rChild=lChild+1");//right child id > than left child id

			double fPivot = pBestSplitPoint[bufferPos].m_fSplitValue;
			double fvalue = pdCurFeaValue[i];
			if(fvalue >= fPivot)
			{
				pInsIdToNodeId[insId] = pRChildId[bufferPos];//right child id
			}
			else
				pInsIdToNodeId[insId] = pLChildId[bufferPos];//left child id
		}
	}

}

__global__ void InsToNewNodeByDefault(TreeNode *pAllTreeNode, int *pInsIdToNodeId, int *pSNIdToBuffId,
										   int *pParentId, int *pLChildId,
										   int preMaxNodeId, int numofIns, int flag_LEAFNODE)
{
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId >= numofIns)//not used threads
		return;

	ErrorChecker(preMaxNodeId, __PRETTY_FUNCTION__, "maxId shoud >= 0");

	int nid = pInsIdToNodeId[nGlobalThreadId];
	if(nid == -1 || nid > preMaxNodeId)//processed node (i.e. leaf node or new node)
		return;
	//newly constructed leaf node
	if(pAllTreeNode[nid].rightChildId == flag_LEAFNODE)
	{
		pInsIdToNodeId[nGlobalThreadId] = -1;
	}
	else
	{
		int bufferPos = pSNIdToBuffId[nid];
		pInsIdToNodeId[nGlobalThreadId] = pLChildId[bufferPos];//by default the instance with unknown feature value going to left child
		ErrorCond(bufferPos != -1, __PRETTY_FUNCTION__, "rChild=lChild+1");
	}

}

__global__ void UpdateNewSplittable(TreeNode *pNewSplittableNode, nodeStat *pNewNodeStat, int *pSNIdToBuffId,
								   	    nodeStat *pSNodeStat, int *pNumofNewNode, int *pBuffIdVec, int *pBuffIdCounter,
								   	    int maxNumofSplittable, int *pnLock)
{
	int numofNewNode = *pNumofNewNode;
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId < 0 || nGlobalThreadId >= numofNewNode)//one thread per splittable node
		printf("Error in InsToNewNode function, thread id=%d\n", nGlobalThreadId);

	ErrorChecker(*pBuffIdCounter == 0, __PRETTY_FUNCTION__, "*pBuffIdCounter == 0");

	int nid = pNewSplittableNode[nGlobalThreadId].nodeId;
	ErrorChecker(nid, __PRETTY_FUNCTION__, "nid");


	bool bLeaveLoop = false;
	while(bLeaveLoop == false)
	{
		//critical region when assigning hash value
		if(atomicExch(pnLock, 1) == 0)
		{
			bool bIsNew = false;
			int bufferPos = AssignHashValue(pSNIdToBuffId, nid, maxNumofSplittable, bIsNew);

			ErrorChecker(bufferPos, __PRETTY_FUNCTION__, "bufferPos");
			pSNodeStat[bufferPos] = pNewNodeStat[nGlobalThreadId];
			if(bIsNew == true)
			{
				int counter = atomicAdd(pBuffIdCounter, 1);
				pBuffIdVec[counter] = bufferPos;
			}
			bLeaveLoop = true;
			atomicExch(pnLock, 0);
		}
	}

}
