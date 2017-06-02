/*
 * SplitAllKernel.cu
 *
 *  Created on: 15 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "DeviceSplitAllKernel.h"
#include "../Memory/gbdtGPUMemManager.h"
#include "../DeviceHashing.h"
#include "../../SharedUtility/CudaMacro.h"

#ifndef testing
#define testing
#endif

/**
 * @brief: compute the base_weight of tree node, also determines if a node is a leaf.
 */
__global__ void ComputeWeight(TreeNode *pAllTreeNode, TreeNode *pNewNode,
							  SplitPoint *pBestSplitPoint, nodeStat *pNewNodeStat, real rt_eps, int flag_LEAFNODE,
							  real lambda, int numofNewNode, bool bLastLevel, int maxNumofSN)
{
	int nGlobalThreadId = GLOBAL_TID();
	if(nGlobalThreadId >= numofNewNode)//one thread per splittable node
		return;

	int nid = pNewNode[nGlobalThreadId].nodeId;
	//new node is no splittable node when it is a leaf.
	if(bLastLevel == true){
		real nodeWeight = (-pNewNodeStat[nGlobalThreadId].sum_gd / (pNewNodeStat[nGlobalThreadId].sum_hess + lambda));
		pAllTreeNode[nid].predValue = nodeWeight;
		return;
	}

	//new node equals to splittable node.
	ECHECKER(nid);

	int snIdPos = nid % maxNumofSN;
	ECHECKER(snIdPos);

	//mark the node as a leaf node if (1) the gain is negative or (2) the tree reaches maximum depth.
	pAllTreeNode[nid].loss = pBestSplitPoint[snIdPos].m_fGain;
	ECHECKER(pNewNodeStat[snIdPos].sum_hess);

	real nodeWeight = (-pNewNodeStat[snIdPos].sum_gd / (pNewNodeStat[snIdPos].sum_hess + lambda));
	pAllTreeNode[nid].base_weight = nodeWeight;
	if(pBestSplitPoint[snIdPos].m_fGain <= rt_eps || bLastLevel == true)
	{
		//printf("gd=%f, hess=%f, lambda=%f; w=%f\n", pSNodeStat[snIdPos].sum_gd, pSNodeStat[snIdPos].sum_hess, lambda, nodeWeight);
		//weight of a leaf node
		pAllTreeNode[nid].predValue = pAllTreeNode[nid].base_weight;
		pAllTreeNode[nid].rightChildId = flag_LEAFNODE;
	}
}

/**
 * @brief: create new nodes and associate new nodes with their parent id
 */
__global__ void CreateNewNode(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, TreeNode *pNewSplittableNode,
							  const SplitPoint *pBestSplitPoint,
								  int *pParentId, int *pLChildId, int *pRChildId,
								  const nodeStat *pLChildStat, const nodeStat *pRChildStat, nodeStat *pNewNodeStat,
								  int *pNumofNode, int *pNumofNewNode,
								  real rt_eps, int nNumofSplittableNode, bool bLastLevel, int maxNumofSN)
{
	//for each splittable node, assign lchild and rchild ids
	int nGlobalThreadId = GLOBAL_TID();
	if(nGlobalThreadId >= nNumofSplittableNode)//one thread per splittable node
		return;

	CONCHECKER(*pNumofNewNode == 0);

	int nid = pSplittableNode[nGlobalThreadId].nodeId;

	ECHECKER(nid);
	int bufferPos = nid % maxNumofSN;

	ECHECKER(bufferPos);
	pAllTreeNode[nid].m_bDefault2Right = false;
	if(!(pBestSplitPoint[bufferPos].m_fGain <= rt_eps || bLastLevel == true))
	{
		int childrenId = atomicAdd(pNumofNode, 2);

		int lchildId = childrenId;
		int rchildId = childrenId + 1;

		//save parent id and child ids
		pParentId[bufferPos] = nid;
		pLChildId[bufferPos] = lchildId;
		pRChildId[bufferPos] = rchildId;
		ECHECKER(pLChildStat[bufferPos].sum_hess);
		ECHECKER(pRChildStat[bufferPos].sum_hess);

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
		leftChild.m_bDefault2Right = false;
		rightChild.featureId = -1;
		rightChild.fSplitValue = -1;
		rightChild.leftChildId = -1;
		rightChild.rightChildId = -1;
		rightChild.loss = -1.0;
		rightChild.m_bDefault2Right = false;

		//they should just be pointers, not new content
		pNewSplittableNode[leftNewNodeId] = leftChild;
		pNewSplittableNode[rightNewNodeId] = rightChild;

		pAllTreeNode[nid].leftChildId = leftChild.nodeId;
		pAllTreeNode[nid].rightChildId = rightChild.nodeId;
		ECHECKER(pBestSplitPoint[bufferPos].m_nFeatureId);

		pAllTreeNode[nid].featureId = pBestSplitPoint[bufferPos].m_nFeatureId;
		pAllTreeNode[nid].fSplitValue = pBestSplitPoint[bufferPos].m_fSplitValue;
		//instances with missing values go to left node by default
		if(pBestSplitPoint[bufferPos].m_bDefault2Right == true)
			pAllTreeNode[nid].m_bDefault2Right = true;

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
	int nGlobalThreadId = GLOBAL_TID();
	if(nGlobalThreadId >= nNumofSplittableNode)//one thread per splittable node
		return;	

	CONCHECKER(*pNumofUniqueFid == 0);

	int fid = pSplittableNode[nGlobalThreadId].featureId;
	int nid = pSplittableNode[nGlobalThreadId].nodeId;
	if(fid == -1 && pAllTreeNode[nid].rightChildId == flag_LEAFNODE)
	{//leaf node should satisfy two conditions at this step
		return;
	}
	ECHECKER(fid);

    int laneid = (threadIdx.x & 31);

    for(int i = 0; i < 32; i++){
    	if (i == laneid){
    		bool bLeaveLoop = false;
    		while(bLeaveLoop == false){
    			//critical region when assigning hash value
    			if(atomicCAS(pnLock, 0, 1) == 0){
    				if(pFeaIdToBuffId[fid % maxNumofUsedFea] == -1){
    					pFeaIdToBuffId[fid % maxNumofUsedFea] = fid;
    					int numofUniqueFid = atomicAdd(pNumofUniqueFid, 1);
    					pUniqueFidVec[numofUniqueFid] = fid;
    				}
    				else if(pFeaIdToBuffId[fid % maxNumofUsedFea] != fid){//a naive solution for resolving conflict
    					for(int k = 0; k < maxNumofUsedFea; k++){
    						if(pFeaIdToBuffId[k] == -1){
    							pFeaIdToBuffId[k] = fid;
    	    					int numofUniqueFid = atomicAdd(pNumofUniqueFid, 1);
    	    					pUniqueFidVec[numofUniqueFid] = fid;
    							break;
    						}
    					}
    				}
    				atomicExch(pnLock, 0);
    				bLeaveLoop = true;
    			}
    		}
        }
    }
}

/**
 * @brief: assign instances (which have non-zero values on the feature of interest) to new nodes
 */
__global__ void InsToNewNode(const TreeNode *pAllTreeNode, const real *pdFeaValue, const int *pInsId,
							 const unsigned int *pFeaStartPos, const int *pNumofKeyValue,
							 const SplitPoint *pBestSplitPoint,
							 const int *pUniqueFidVec, const int *pNumofUniqueFid,
							 const int *pParentId, const int *pLChildId, const int *pRChildId,
								 int preMaxNodeId, int numofFea, int *pInsIdToNodeId, int numofIns, int flag_LEAFNODE,
								 const int maxSN)
{
	int numofUniqueFid = *pNumofUniqueFid;
	int feaId = blockIdx.z;
	CONCHECKER(feaId < numofUniqueFid);

	int ufid = pUniqueFidVec[feaId];

	ECHECKER(ufid);
	ECHECKER(numofFea - ufid);

	int nNumofPair = pNumofKeyValue[ufid];//number of feature values in the form of (ins_id, fvalue)
	int perFeaTid = GLOBAL_TID();
	if(perFeaTid >= nNumofPair)//one thread per feaValue
		return;

	//for each instance that has value on the feature
	unsigned int curFeaStartPos = pFeaStartPos[ufid];//this start pos is never changed (i.e. always the same as the original)
	const real *pdCurFeaValue = pdFeaValue + curFeaStartPos;//fvalue start pos in the global memory
	const int *pCurFeaInsId = pInsId + curFeaStartPos;//ins_id of this fea start pos in the global memory

	int insId = pCurFeaInsId[perFeaTid];

	ECHECKER(numofIns - insId);
	ECHECKER(insId);

	int nid = pInsIdToNodeId[insId];
	ECHECKER(nid);

	if(nid > preMaxNodeId)//new node ids. This is possible because here each thread 
						  //corresponds to a feature value, and hence duplication may occur.
		return;
	if(pAllTreeNode[nid].rightChildId == flag_LEAFNODE)//leaf node
		return;

	int bufferPos = nid % maxSN;

	ECHECKER(bufferPos);

	int fid = pBestSplitPoint[bufferPos].m_nFeatureId;
	if(fid != ufid)//this feature is not the splitting feature for the instance.
		return;

	if(nid != pParentId[bufferPos]){//node doesn't need to split (leaf node or new node)
		if(pAllTreeNode[nid].rightChildId != flag_LEAFNODE){
			ECHECKER(preMaxNodeId - nid);
			return;
		}
		CONCHECKER(pAllTreeNode[nid].rightChildId == flag_LEAFNODE);
		return;
	}

	if(nid == pParentId[bufferPos]){//internal node (needs to split)
			CONCHECKER(pRChildId[bufferPos] == pLChildId[bufferPos] + 1);//right child id > than left child id
			CONCHECKER(pAllTreeNode[nid].rightChildId != flag_LEAFNODE);
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

__global__ void InsToNewNodeByDefault(TreeNode *pAllTreeNode, int *pInsIdToNodeId,
									  int *pParentId, int *pLChildId, int *pRChildId,
									  int preMaxNodeId, int numofIns, int flag_LEAFNODE,
									  const SplitPoint *pBestSplitPoint, const int maxSN)
{
	int nGlobalThreadId = GLOBAL_TID();
	if(nGlobalThreadId >= numofIns)//not used threads
		return;

	ECHECKER(preMaxNodeId);

	int nid = pInsIdToNodeId[nGlobalThreadId];
	ECHECKER(nid);
	if(nid > preMaxNodeId)//processed node
		return;

	if(pAllTreeNode[nid].rightChildId == flag_LEAFNODE)//leaf node
		return;
	else
	{
//		printf("ins to new node by default: ################## nid=%d, maxNid=%d, rcid=%d, flag=%d\n", nid, preMaxNodeId, pAllTreeNode[nid].rightChildId, flag_LEAFNODE);
		int bufferPos = nid % maxSN; //pSNIdToBuffId[nid];
		if(pBestSplitPoint[bufferPos].m_bDefault2Right == false)
			pInsIdToNodeId[nGlobalThreadId] = pLChildId[bufferPos];//by default the instance with unknown feature value going to left child
		else
			pInsIdToNodeId[nGlobalThreadId] = pRChildId[bufferPos];
		CONCHECKER(bufferPos != -1);
//		atomicAdd(numInsL + bufferPos, 1);
	}

}

__global__ void UpdateNewSplittable(TreeNode *pNewSplittableNode, nodeStat *pNewNodeStat,
								   	    nodeStat *pSNodeStat, int *pNumofNewNode, int *pPartitionId2SNPos, int *pPartitionCounter,
								   	    int maxNumofSplittable, int *pnLock, int preMaxNodeId)
{
	int numofNewNode = *pNumofNewNode;
	int nGlobalThreadId = GLOBAL_TID();
	if(nGlobalThreadId >= numofNewNode)//one thread per splittable node
		return;

	CONCHECKER(*pPartitionCounter == 0);

	int nid = pNewSplittableNode[nGlobalThreadId].nodeId;
	ECHECKER(nid);


	bool bIsNew = false;
	int snPos = nid % maxNumofSplittable;

	ECHECKER(snPos);
	pSNodeStat[snPos] = pNewNodeStat[nGlobalThreadId];

	ECHECKER(nid - preMaxNodeId - 1);
	pPartitionId2SNPos[nid - preMaxNodeId - 1] = snPos;

	//for computing node size
	pNewSplittableNode[nGlobalThreadId].numIns = pNewNodeStat[nGlobalThreadId].sum_hess;//Will this have problems? sum_hess is count on fvalue != 0, while numIns may be bigger.
}
