/*
 * DeviceSplitAll.h
 *
 *  Created on: 15 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITALL_H_
#define DEVICESPLITALL_H_

#include "../../pureHost/UpdateOps/NodeStat.h"
#include "../../pureHost/UpdateOps/SplitPoint.h"
#include "../../pureHost/BaseClasses/BaseSplitter.h"


typedef double float_point;

__global__ void ComputeWeight(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, int *pSNIdToBufferId,
		  	  	  	  	  	  	  SplitPoint *pBestSplitPoint, nodeStat *pSNodeStat, float_point rt_eps, int flag_LEAFNODE,
		  	  	  	  	  	  	  float_point lambda, int numofSplittableNode, bool bLastLevel);

__global__ void CreateNewNode(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, TreeNode *pNewSplittableNode,
		  	  	  	  	  	  	  int *pSNIdToBufferId, SplitPoint *pBestSplitPoint,
		  	  	  	  	  	  	  int *pParentId, int *pLChildId, int *pRChildId,
		  	  	  	  	  	  	  nodeStat *pLChildStat, nodeStat *pRChildStat, nodeStat *pNewNodeStat,
		  	  	  	  	  	  	  int *m_nNumofNode, int *pNumofNewNode,
		  	  	  	  	  	  	  float_point rt_eps, int nNumofSplittableNode, bool bLastLevel);


__global__ void GetUniqueFid(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, int nNumofSplittableNode,
		 	 	 	 	 	 	 int *pFeaIdToBuffId, int *pUniqueFidVec,int *pNumofUniqueFid,
		 	 	 	 	 	 	 int maxNumofUsedFea, int flag_LEAFNODE);

__global__ void InsToNewNode(TreeNode *pAllTreeNode, float_point *pdFeaValue, int *pInsId,
		 	 	 	 	 	 	 long long *pFeaStartPos, int *pNumofKeyValue,
		 	 	 	 	 	 	 int *pInsIdToNodeId, int *pSNIdToBuffId, SplitPoint *pBestSplitPoint,
		 	 	 	 	 	 	 int *pUniqueFidVec, int *pNumofUniqueFid,
		 	 	 	 	 	 	 int *pParentId, int *pLChildId, int *pRChildId,
		 	 	 	 	 	 	 int preMaxNodeId, int numofFea, int numofIns, int flag_LEAFNODE);

__global__ void InsToNewNodeByDefault(TreeNode *pAllTreeNode, int *pInsIdToNodeId, int *pSNIdToBuffId,
		   	   	   	   	   	   	   	   	   int *pParentId, int *pLChildId,
		   	   	   	   	   	   	   	   	   int preMaxNodeId, int numofIns, int flag_LEAFNODE);

__global__ void UpdateNewSplittable(TreeNode *pNewSplittableNode, nodeStat *pNewNodeStat, int *pSNIdToBuffId,
		   	   	   	   	   	   	   	    nodeStat *pSNodeStat, int *pNumofNewNode, int *pBuffIdVec,
		   	   	   	   	   	   	   	    int *pBuffIdCounter, int maxNumofSplittable);

#endif /* DEVICESPLITALL_H_ */
