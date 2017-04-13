/*
 * DeviceSplitAll.h
 *
 *  Created on: 15 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITALL_H_
#define DEVICESPLITALL_H_

#include "../../Host/UpdateOps/SplitPoint.h"
#include "../../DeviceHost/BaseClasses/BaseSplitter.h"
#include "../../DeviceHost/DefineConst.h"
#include "../../DeviceHost/NodeStat.h"


__global__ void ComputeWeight(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, const int *pSNIdToBufferId,
		  	  	  	  	  	  	  SplitPoint *pBestSplitPoint, nodeStat *pSNodeStat, float_point rt_eps, int flag_LEAFNODE,
		  	  	  	  	  	  	  float_point lambda, int numofSplittableNode, bool bLastLevel, int maxNumofSplittableNode);

__global__ void CreateNewNode(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, TreeNode *pNewSplittableNode,
								  const int *pSNIdToBufferId, const SplitPoint *pBestSplitPoint,
		  	  	  	  	  	  	  int *pParentId, int *pLChildId, int *pRChildId,
		  	  	  	  	  	  	  const nodeStat *pLChildStat, const nodeStat *pRChildStat, nodeStat *pNewNodeStat,
		  	  	  	  	  	  	  int *m_nNumofNode, int *pNumofNewNode,
		  	  	  	  	  	  	  float_point rt_eps, int nNumofSplittableNode, bool bLastLevel, int maxNumofSplittableNode);


__global__ void GetUniqueFid(TreeNode *pAllTreeNode, TreeNode *pSplittableNode, int nNumofSplittableNode,
		 	 	 	 	 	 	 int *pFeaIdToBuffId, int *pUniqueFidVec,int *pNumofUniqueFid,
		 	 	 	 	 	 	 int maxNumofUsedFea, int flag_LEAFNODE, int *pnLock);

__global__ void InsToNewNode(const TreeNode *pAllTreeNode, const float_point *pdFeaValue, const int *pInsId,
		 	 	 	 	 	 const long long *pFeaStartPos, const int *pNumofKeyValue,
		 	 	 	 	 	 const int *pSNIdToBuffId, const SplitPoint *pBestSplitPoint,
		 	 	 	 	 	 const int *pUniqueFidVec, const int *pNumofUniqueFid,
		 	 	 	 	 	 const int *pParentId, const int *pLChildId, const int *pRChildId,
		 	 	 	 	 	 int preMaxNodeId, int numofFea, int *pInsIdToNodeId, int numofIns, int flag_LEAFNODE);

__global__ void InsToNewNodeByDefault(TreeNode *pAllTreeNode, int *pInsIdToNodeId, const int *pSNIdToBuffId,
		   	   	   	   	   	   	   	   	   int *pParentId, int *pLChildId,
		   	   	   	   	   	   	   	   	   int preMaxNodeId, int numofIns, int flag_LEAFNODE);

__global__ void UpdateNewSplittable(TreeNode *pNewSplittableNode, nodeStat *pNewNodeStat, int *pSNIdToBuffId,
		   	   	   	   	   	   	   	    nodeStat *pSNodeStat, int *pNumofNewNode, int *pBuffIdVec,
		   	   	   	   	   	   	   	    int *pBuffIdCounter, int maxNumofSplittable, int *pnLock);

#endif /* DEVICESPLITALL_H_ */
