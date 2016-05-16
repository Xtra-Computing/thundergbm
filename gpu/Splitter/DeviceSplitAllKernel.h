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
		  	  	  	  	  	  	  int *m_nNumofNode,
		  	  	  	  	  	  	  float_point rt_eps, int nNumofSplittableNode, bool bLastLevel);

#endif /* DEVICESPLITALL_H_ */
