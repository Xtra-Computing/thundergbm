/*
 * gbdtGPUMemManager.cu
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <helper_cuda.h>

#include "gbdtGPUMemManager.h"
#include "../../SharedUtility/CudaMacro.h"

//memory for instances (key on feature id)
int *GBDTGPUMemManager::m_pDInsId = NULL;				//all the instance ids for each key-value pair
int *GBDTGPUMemManager::m_pDNumofKeyValue = NULL;		//the number of key-value pairs of each feature
uint *GBDTGPUMemManager::m_pFeaStartPos = NULL;	//start key-value position of each feature

//memory for instances (key on instance id)
int *GBDTGPUMemManager::m_pDFeaId = NULL;				//all the feature ids for every instance
real *GBDTGPUMemManager::m_pdDInsValue = NULL;	//all the feature values for every instance
int *GBDTGPUMemManager::m_pDNumofFea = NULL;			//the number of features for each instance
uint *GBDTGPUMemManager::m_pInsStartPos = NULL;	//the start position of each instance
//with bag info
real *GBDTGPUMemManager::m_pdDenseInsEachBag = NULL;
int *GBDTGPUMemManager::m_pHashFeaIdToDenseInsPosBag = NULL;	//hash map for used feature ids of all trees to the dense instance position
int *GBDTGPUMemManager::m_pSortedUsedFeaIdBag = NULL;			//sorted used feature ids
TreeNode *GBDTGPUMemManager::m_pAllTreeEachBag = NULL;
int *GBDTGPUMemManager::m_pNumofTreeLearntEachBag_h = NULL;
int *GBDTGPUMemManager::m_pStartPosOfEachTreeEachBag = NULL;	//the start position of each tree in the memory
int *GBDTGPUMemManager::m_pNumofNodeEachTreeEachBag = NULL;	//the number of nodes of each tree

//memory for prediction
int GBDTGPUMemManager::m_maxUsedFeaInATree = -1;		//maximum number of used features in all the trees

unsigned int GBDTGPUMemManager::m_numFeaValue = 0;
int GBDTGPUMemManager::m_numofIns = -1;
int GBDTGPUMemManager::m_numofFea = -1;

/**
 * @brief: allocate memory for training instances
 */
void GBDTGPUMemManager::mallocForTrainingIns(int nTotalNumofValue, int numofIns, int numofFeature){
	PROCESS_ERROR(nTotalNumofValue > 0);
	PROCESS_ERROR(numofFeature > 0);
	PROCESS_ERROR(numofIns > 0);
	m_numFeaValue = nTotalNumofValue;
	m_numofIns = numofIns;
	m_numofFea = numofFeature;

	//memory for instances (key on feature id)
	checkCudaErrors(cudaMalloc((void**)&m_pDInsId, sizeof(int) * m_numFeaValue));
	checkCudaErrors(cudaMalloc((void**)&m_pDNumofKeyValue, sizeof(int) * m_numofFea));
	checkCudaErrors(cudaMalloc((void**)&m_pFeaStartPos, sizeof(uint) * m_numofFea));
}

void GBDTGPUMemManager::freeMemForTrainingIns(){
	//memory for instances (key on feature id)
	checkCudaErrors(cudaFree(m_pDInsId));
	checkCudaErrors(cudaFree(m_pDNumofKeyValue));
	checkCudaErrors(cudaFree(m_pFeaStartPos));
}

/*
 * @brief for testing instances
 */
void GBDTGPUMemManager::mallocForTestingIns(int nTotalNumofValue, int numofIns, int numofFeature, int numBag,
											int numTreeABag, int maxNumNode){
	PROCESS_ERROR(nTotalNumofValue > 0);
	PROCESS_ERROR(numofFeature > 0);
	PROCESS_ERROR(numofIns > 0);
	m_numFeaValue = nTotalNumofValue;
	m_numofIns = numofIns;
	m_numofFea = numofFeature;
	//memory for instances (key on instance id); for prediction
	checkCudaErrors(cudaMallocManaged((void**)&m_pDFeaId, sizeof(int) * m_numFeaValue));
	checkCudaErrors(cudaMallocManaged((void**)&m_pdDInsValue, sizeof(real) * m_numFeaValue));
	checkCudaErrors(cudaMallocManaged((void**)&m_pDNumofFea, sizeof(int) * m_numofIns));
	checkCudaErrors(cudaMallocManaged((void**)&m_pInsStartPos, sizeof(uint) * m_numofIns));

	//for bags
	checkCudaErrors(cudaMallocManaged((void**)&m_pdDenseInsEachBag, sizeof(real) * m_maxUsedFeaInATree * m_numofIns * numBag));
	checkCudaErrors(cudaMallocManaged((void**)&m_pHashFeaIdToDenseInsPosBag, sizeof(int) * m_maxUsedFeaInATree * numBag));
	checkCudaErrors(cudaMemset(m_pHashFeaIdToDenseInsPosBag, -1, sizeof(int) * m_maxUsedFeaInATree * numBag));
	checkCudaErrors(cudaMallocManaged((void**)&m_pSortedUsedFeaIdBag, sizeof(int) * m_maxUsedFeaInATree * numBag));
	checkCudaErrors(cudaMallocManaged((void**)&m_pAllTreeEachBag, sizeof(TreeNode) * numTreeABag * maxNumNode * numBag));
	//memory set for all tree nodes
	TreeNode *pAllTreeNodeHost = new TreeNode[numTreeABag * maxNumNode * numBag];
	checkCudaErrors(cudaMemcpy(m_pAllTreeEachBag, pAllTreeNodeHost, sizeof(TreeNode) * numTreeABag * maxNumNode * numBag, cudaMemcpyHostToDevice));
	delete[] pAllTreeNodeHost;
	checkCudaErrors(cudaMallocManaged((void**)&m_pStartPosOfEachTreeEachBag, sizeof(int) * numTreeABag * numBag));
	checkCudaErrors(cudaMemset(m_pStartPosOfEachTreeEachBag, -1, sizeof(int) * numTreeABag * numBag));
	//for individual tree
	checkCudaErrors(cudaMallocManaged((void**)&m_pNumofNodeEachTreeEachBag, sizeof(int) * numTreeABag * numBag));
	checkCudaErrors(cudaMemset(m_pNumofNodeEachTreeEachBag, 0, sizeof(int) * numTreeABag * numBag));

	m_pNumofTreeLearntEachBag_h = new int[numBag];
	memset(m_pNumofTreeLearntEachBag_h, 0, sizeof(int) * numBag);
}

/**
 * @brief: free memory for testing instances
 */
void GBDTGPUMemManager::freeMemForTestingIns(){
	//memory for instances (key on instance id); for prediction
	checkCudaErrors(cudaFree(m_pDFeaId));
	checkCudaErrors(cudaFree(m_pdDInsValue));
	checkCudaErrors(cudaFree(m_pDNumofFea));
	checkCudaErrors(cudaFree(m_pInsStartPos));
	checkCudaErrors(cudaFree(m_pdDenseInsEachBag));
	checkCudaErrors(cudaFree(m_pHashFeaIdToDenseInsPosBag));
	checkCudaErrors(cudaFree(m_pSortedUsedFeaIdBag));
	checkCudaErrors(cudaFree(m_pAllTreeEachBag));
	checkCudaErrors(cudaFree(m_pStartPosOfEachTreeEachBag));
	checkCudaErrors(cudaFree(m_pNumofNodeEachTreeEachBag));
	delete []m_pNumofTreeLearntEachBag_h;
}
