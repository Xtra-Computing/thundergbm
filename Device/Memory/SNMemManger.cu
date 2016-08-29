/*
 * SplitNodeGPUMemManger.cu
 *
 *  Created on: 16/05/2016
 *      Author: zeyi
 */

#include <helper_cuda.h>
#include "SNMemManager.h"
#include "../../DeviceHost/MyAssert.h"
#include "gpuMemManager.h"


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
int *SNGPUManager::m_pCurNumofNode_d = NULL;
int *SNGPUManager::m_pNumofNewNode = NULL;

//for used features
int SNGPUManager::m_maxNumofUsedFea = -1;	//for reserving GPU memory; maximum number of used features in a tree
int *SNGPUManager::m_pFeaIdToBuffId = NULL;//(require memset!) map feature id to buffer id
int *SNGPUManager::m_pUniqueFeaIdVec = NULL;	//store all the used feature ids
int *SNGPUManager::m_pNumofUniqueFeaId = NULL;//(require memset!)store the number of unique feature ids

//host memory for reset purposes
TreeNode *SNGPUManager::m_pTreeNodeHost = NULL;

/**
 * @brief: reserve memory for the tree
 */
void SNGPUManager::allocMemForTree(int maxNumofNode)
{
	PROCESS_ERROR(maxNumofNode > 0);
	m_maxNumofNode = maxNumofNode;
	checkCudaErrors(cudaMalloc((void**)&m_pTreeNode, sizeof(TreeNode) * m_maxNumofNode));
	checkCudaErrors(cudaMalloc((void**)&m_pCurNumofNode_d, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&m_pNumofNewNode, sizeof(int)));

	//for reseting memory for the next tree
	m_pTreeNodeHost = new TreeNode[m_maxNumofNode];
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

/**
 * @brief: memory for splitAll process
 */
void SNGPUManager::allocMemForUsedFea(int nMaxNumofUsedFeature)
{
	PROCESS_ERROR(nMaxNumofUsedFeature > 0);
	m_maxNumofUsedFea = nMaxNumofUsedFeature;
	//map splittable node to buffer id
	checkCudaErrors(cudaMalloc((void**)&m_pFeaIdToBuffId, sizeof(int) * m_maxNumofUsedFea));
	checkCudaErrors(cudaMalloc((void**)&m_pUniqueFeaIdVec, sizeof(int) * m_maxNumofUsedFea));
	checkCudaErrors(cudaMalloc((void**)&m_pNumofUniqueFeaId, sizeof(int)));
}
