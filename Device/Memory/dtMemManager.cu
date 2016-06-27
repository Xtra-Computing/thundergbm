/*
 * dtMemManager.cu
 *
 *  Created on: 25 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <helper_cuda.h>
#include "dtMemManager.h"
#include "../../DeviceHost/MyAssert.h"

TreeNode *DTGPUMemManager::m_pAllTree = NULL;	//all the decision trees
int *DTGPUMemManager::m_pNumofNodeEachTree = NULL;	//the number of nodes of each tree
int *DTGPUMemManager::m_pStartPosOfEachTree = NULL;	//the start position of each tree in the memory
int DTGPUMemManager::m_numofTree = -1;				//total number of trees
int DTGPUMemManager::m_numofTreeLearnt = 0;

/**
 * @brief: allocate memory for trees
 */
void DTGPUMemManager::allocMemForTrees(int numofTree, int maxNumofNodePerTree)
{
	PROCESS_ERROR(numofTree > 0);
	PROCESS_ERROR(maxNumofNodePerTree > 0);
	m_numofTree = numofTree;
	m_numofTreeLearnt = 0;

	checkCudaErrors(cudaMalloc((void**)&m_pNumofNodeEachTree, sizeof(int) * m_numofTree));
	checkCudaErrors(cudaMemset(m_pNumofNodeEachTree, 0, sizeof(int) * m_numofTree));
	checkCudaErrors(cudaMalloc((void**)&m_pStartPosOfEachTree, sizeof(int) * m_numofTree));
	checkCudaErrors(cudaMemset(m_pStartPosOfEachTree, -1, sizeof(int) * m_numofTree));
	checkCudaErrors(cudaMalloc((void**)&m_pAllTree, sizeof(TreeNode) * m_numofTree * maxNumofNodePerTree));
}
