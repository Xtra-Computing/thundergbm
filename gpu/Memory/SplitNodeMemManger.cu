/*
 * SplitNodeGPUMemManger.cu
 *
 *  Created on: 16/05/2016
 *      Author: zeyi
 */

#include <helper_cuda.h>
#include "SplitNodeMemManager.h"
#include "../../pureHost/MyAssert.h"


//memory for the tree
TreeNode *SNGPUManager::m_pTreeNode = NULL;//reserve memory for all nodes of the tree
int SNGPUManager::m_maxNumofNode = -1;

//memory for parent node to children ids
int *SNGPUManager::m_pParentId = NULL;
int *SNGPUManager::m_pLeftChildId = NULL;
int *SNGPUManager::m_pRightChildId = NULL;

//memory for new node statistics
nodeStat *SNGPUManager::m_pNewNodeStat = NULL;
TreeNode *SNGPUManager::m_pNewSplittableNode = NULL;

//current numof nodes
int *SNGPUManager::m_pCurNumofNode = NULL;

/**
 * @brief: reserve memory for the tree
 */
void SNGPUManager::allocMemForTree(int maxNumofNode)
{
	PROCESS_ERROR(maxNumofNode > 0);
	m_maxNumofNode = maxNumofNode;
	checkCudaErrors(cudaMalloc((void**)&m_pTreeNode, sizeof(TreeNode) * m_maxNumofNode));
	checkCudaErrors(cudaMalloc((void**)&m_pCurNumofNode, sizeof(int)));

}

/**
 * @brief: reserve memory for the parent children id mapping
 */
void SNGPUManager::allocMemForParenChildIdMapping(int maxNumofSplittable)
{
	PROCESS_ERROR(maxNumofSplittable > 0);
	checkCudaErrors(cudaMalloc((void**)&m_pParentId, sizeof(int) * maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pLeftChildId, sizeof(int) * maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pRightChildId, sizeof(int) * maxNumofSplittable));
}

/**
 * @brief: reserve memory for new node statistics
 */
void SNGPUManager::allocMemForNewNode(int maxNumofSplittable)
{
	PROCESS_ERROR(maxNumofSplittable > 0);
	checkCudaErrors(cudaMalloc((void**)&m_pNewNodeStat, sizeof(nodeStat) * maxNumofSplittable));
	checkCudaErrors(cudaMalloc((void**)&m_pNewSplittableNode, sizeof(TreeNode) * maxNumofSplittable));
}
