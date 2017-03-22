/*
 * SplitAllKernel.cu
 *
 *  Created on: 15 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "DeviceSplitAllKernel.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../DeviceHashing.h"
#include "../ErrorChecker.h"

#ifndef testing
#define testing
#endif

/**
 * @brief: compute the base_weight of tree node, also determines if a node is a leaf.
 */
__global__ void ComputeWeight(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, const int *pSNIdToBufferId,
								  SplitPoint *pBestSplitPoint, nodeStat *pSNodeStat, float_point rt_eps, int flag_LEAFNODE,
								  float_point lambda, int numofSplittableNode, bool bLastLevel, int maxNumofSplittableNode)
{
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId >= numofSplittableNode)//one thread per splittable node
		return;

	int nid = pSplittableNode[nGlobalThreadId].nodeId;
	ErrorChecker(nid, __PRETTY_FUNCTION__, "nid");

	int bufferPos = GetBufferId(pSNIdToBufferId, nid, maxNumofSplittableNode);
	if(bufferPos != nid % maxNumofSplittableNode)
		printf("ohhhhhhhhhhhhhhhhhhhhhhhhhhhh: bufferPos is unexpected\n");

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
								  float_point rt_eps, int nNumofSplittableNode, bool bLastLevel, int maxNumofSplittableNode)
{
	//for each splittable node, assign lchild and rchild ids
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId >= nNumofSplittableNode)//one thread per splittable node
		return;

	ErrorChecker(*pNumofNewNode == 0, __PRETTY_FUNCTION__, "*pNumofNewNode == 0");

	int nid = pSplittableNode[nGlobalThreadId].nodeId;

	ErrorChecker(nid, __PRETTY_FUNCTION__, "nid");
//	int bufferPos = pSNIdToBufferId[nid];//#########
	int bufferPos = GetBufferId(pSNIdToBufferId, nid, maxNumofSplittableNode);
	if(bufferPos != nid % maxNumofSplittableNode)
		printf("oh shit you ####################################\n");
//	printf("splitting node %d, buffPos is %d, tid=%d\n", nid, bufferPos, nGlobalThreadId);
	ErrorChecker(bufferPos, __PRETTY_FUNCTION__, "bufferPos");

	if(!(pBestSplitPoint[bufferPos].m_fGain <= rt_eps || bLastLevel == true))
	{
		int childrenId = atomicAdd(pNumofNode, 2);

		int lchildId = childrenId;
		int rchildId = childrenId + 1;

		//save parent id and child ids
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

		//init the nodes
		leftChild.featureId = -1;
		leftChild.fSplitValue = -1;
		leftChild.leftChildId = -1;
		leftChild.rightChildId = -1;
		leftChild.loss = -1.0;
		rightChild.featureId = -1;
		rightChild.fSplitValue = -1;
		rightChild.leftChildId = -1;
		rightChild.rightChildId = -1;
		rightChild.loss = -1.0;

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
	if(nGlobalThreadId >= nNumofSplittableNode)//one thread per splittable node
		return;	

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
//		printf("lock is %d\n", *pnLock);
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
								 int *pInsIdToNodeId, const int *pSNIdToBuffId, SplitPoint *pBestSplitPoint,
								 int *pUniqueFidVec, int *pNumofUniqueFid,
								 int *pParentId, int *pLChildId, int *pRChildId,
								 int preMaxNodeId, int numofFea, int numofIns, int flag_LEAFNODE)
{
	int numofUniqueFid = *pNumofUniqueFid;
	int feaId = blockIdx.z;
#ifdef testing
	ErrorCond(feaId < numofUniqueFid, __PRETTY_FUNCTION__, "ufid");
#endif
	int ufid = pUniqueFidVec[feaId];

#ifdef testing
	ErrorChecker(ufid, __PRETTY_FUNCTION__, "ufid");
	ErrorChecker(numofFea - ufid, __PRETTY_FUNCTION__, "numofFea - ufid");
#endif

	int nNumofPair = pNumofKeyValue[ufid];//number of feature values in the form of (ins_id, fvalue)
	int perFeaTid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(perFeaTid >= nNumofPair)//one thread per feaValue
		return;

	//for each instance that has value on the feature
	long long curFeaStartPos = pFeaStartPos[ufid];
	float_point *pdCurFeaValue = pdFeaValue + curFeaStartPos;//fvalue start pos in the global memory
	int *pCurFeaInsId = pInsId + curFeaStartPos;//ins_id of this fea start pos in the global memory

	int insId = pCurFeaInsId[perFeaTid];

#ifdef testing
	ErrorChecker(numofIns - insId, __PRETTY_FUNCTION__, "numofIns - insId");
	ErrorChecker(insId, __PRETTY_FUNCTION__, "insId");
#endif
	int nid = pInsIdToNodeId[insId];

	if(nid < 0)//leaf node
		return;

	if(nid > preMaxNodeId)//new node ids. This is possible because here each thread 
						  //corresponds to a feature value, and hence duplication may occur.
		return;

	int bufferPos = pSNIdToBuffId[nid];

#ifdef testing
	ErrorChecker(bufferPos, __PRETTY_FUNCTION__, "bufferPos");
#endif
	int fid = pBestSplitPoint[bufferPos].m_nFeatureId;
	if(fid != ufid)//this feature is not the splitting feature for the instance.
		return;

	if(nid != pParentId[bufferPos])//node doesn't need to split (leaf node or new node)
	{
		printf("nid=%d, pid=%d ######################\n", nid, pParentId[bufferPos]);
		if(pAllTreeNode[nid].rightChildId != flag_LEAFNODE)
		{
#ifdef testing
			ErrorChecker(preMaxNodeId - nid, __PRETTY_FUNCTION__, "preMaxNodeId - nid");
#endif
			return;
		}
#ifdef testing
		ErrorCond(pAllTreeNode[nid].rightChildId == flag_LEAFNODE, __PRETTY_FUNCTION__, "pAllTreeNode[nid].rightChildId == flag_LEAFNODE");
#endif
		return;
	}

	if(nid == pParentId[bufferPos]){//internal node (needs to split)
#ifdef testing
		ErrorCond(pRChildId[bufferPos] == pLChildId[bufferPos] + 1, __PRETTY_FUNCTION__, "rChild=lChild+1");//right child id > than left child id
#endif
		if(pAllTreeNode[nid].rightChildId == flag_LEAFNODE)
			printf("Are you kidding me????????????????\n");
		double fPivot = pBestSplitPoint[bufferPos].m_fSplitValue;
		double fvalue = pdCurFeaValue[perFeaTid];

		if(fvalue >= fPivot){
			pInsIdToNodeId[insId] = pRChildId[bufferPos];//right child id
//			atomicAdd(numInsR + bufferPos, 1);//increase numIns in right child
		}
		else{
			pInsIdToNodeId[insId] = pLChildId[bufferPos];//left child id
//			atomicAdd(numInsL + bufferPos, 1);
		}
	}
}

__global__ void InsToNewNodeByDefault(TreeNode *pAllTreeNode, int *pInsIdToNodeId, const int *pSNIdToBuffId,
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
		printf("ins to new node by default: ################## nid=%d, maxNid=%d, rcid=%d, flag=%d\n", nid, preMaxNodeId, pAllTreeNode[nid].rightChildId, flag_LEAFNODE);
		int bufferPos = pSNIdToBuffId[nid];
		//if(pInsIdToNodeId[nGlobalThreadId] * 2 + 1 != pLChildId[bufferPos])
		pInsIdToNodeId[nGlobalThreadId] = pLChildId[bufferPos];//by default the instance with unknown feature value going to left child
		ErrorCond(bufferPos != -1, __PRETTY_FUNCTION__, "rChild=lChild+1");
//		atomicAdd(numInsL + bufferPos, 1);
	}

}

__global__ void UpdateNewSplittable(TreeNode *pNewSplittableNode, nodeStat *pNewNodeStat, int *pSNIdToBuffId,
								   	    nodeStat *pSNodeStat, int *pNumofNewNode, int *pBuffIdVec, int *pBuffIdCounter,
								   	    int maxNumofSplittable, int *pnLock)
{
	int numofNewNode = *pNumofNewNode;
	int nGlobalThreadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(nGlobalThreadId >= numofNewNode)//one thread per splittable node
		return;

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
			if(bufferPos != nid % maxNumofSplittable)
				printf("oh shit ###################################\n");

			ErrorChecker(bufferPos, __PRETTY_FUNCTION__, "bufferPos");
			pSNodeStat[bufferPos] = pNewNodeStat[nGlobalThreadId];
			if(bIsNew == true)
			{
				int counter = atomicAdd(pBuffIdCounter, 1);
				ErrorChecker(counter, __PRETTY_FUNCTION__, "counter");
				pBuffIdVec[counter] = bufferPos;
			}
			bLeaveLoop = true;
			atomicExch(pnLock, 0);
		}
	}
	//for computing node size
	pNewSplittableNode[nGlobalThreadId].numIns = pNewNodeStat[nGlobalThreadId].sum_hess;//Will this have problems? sum_hess is count on fvalue != 0, while numIns may be bigger.
//	printf("nid=%d, numofIns=%d\n", nid, pNewSplittableNode[nGlobalThreadId].numIns);
}
