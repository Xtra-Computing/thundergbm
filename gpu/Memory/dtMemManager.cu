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

/**
 * @brief: allocate memory for trees
 */
void DTGPUMemManager::allocMemForTrees(int numofTree, int maxNumofNodePerTree)
{
	PROCESS_ERROR(numofTree > 0);
	PROCESS_ERROR(maxNumofNodePerTree > 0);
	m_numofTree = numofTree;

	checkCudaErrors(cudaMalloc((void**)&m_pNumofNodeEachTree, sizeof(int) * m_numofTree));
	checkCudaErrors(cudaMalloc((void**)&m_pStartPosOfEachTree, sizeof(int) * m_numofTree));
	checkCudaErrors(cudaMalloc((void**)&m_pAllTree, sizeof(TreeNode) * m_numofTree * maxNumofNodePerTree));
}
