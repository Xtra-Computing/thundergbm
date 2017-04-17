/*
 * BagManager.cu
 *
 *  Created on: 8 Aug 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "BagManager.h"
#include "BagBuilder.h"
#include "../Memory/gpuMemManager.h"
#include "../../DeviceHost/MyAssert.h"
#include "../../SharedUtility/KernelConf.h"
#include "../../SharedUtility/CudaMacro.h"

int *BagManager::m_pInsWeight = NULL;
int BagManager::m_numBag = -1;
int BagManager::m_numIns = -1;
int BagManager::m_numFea = -1;
long long BagManager::m_numFeaValue = -1;

//tree information
int BagManager::m_numTreeEachBag = -1;
int BagManager::m_maxNumNode = -1;
int BagManager::m_maxNumSplittable = -1;
int BagManager::m_maxTreeDepth = -1;

//device memory
cudaStream_t *BagManager::m_pStream = NULL;
TreeNode *BagManager::m_pSNBufferEachBag = NULL;
int *BagManager::m_pInsIdToNodeIdEachBag = NULL;	//instance to node id
int *BagManager::m_pInsWeight_d = NULL;
TreeNode *BagManager::m_pAllTreeEachBag = NULL;
int *BagManager::m_pNumofTreeLearntEachBag_h = NULL;

//for gd/hessian computation
//memory for initialisation
float_point *BagManager::m_pGDBlockSumEachBag = NULL;	//for initialising root node
float_point *BagManager::m_pHessBlockSumEachBag = NULL;//for initialising root node
int BagManager::m_numBlockForBlockSum = -1;
float_point *BagManager::m_pPredBufferEachBag = NULL;
float_point *BagManager::m_pdDenseInsEachBag = NULL;
float_point *BagManager::m_pdTrueTargetValueEachBag = NULL;	//true target value of each instance
float_point *BagManager::m_pTargetValueEachBag = NULL;	//predicted target value of each instance
float_point *BagManager::m_pInsGradEachBag = NULL;
float_point *BagManager::m_pInsHessEachBag = NULL;
float_point *BagManager::m_pGDEachFvalueEachBag = NULL;		//gd of each feature value
float_point *BagManager::m_pHessEachFvalueEachBag = NULL;	//hessian of each feature value
float_point *BagManager::m_pDenseFValueEachBag = NULL;		//feature values of consideration (use for computing the split?)
float_point *BagManager::m_pGDPrefixSumEachBag = NULL;		//gd prefix sum for each feature
float_point *BagManager::m_pHessPrefixSumEachBag = NULL;	//hessian prefix sum for each feature
float_point *BagManager::m_pGainEachFvalueEachBag = NULL;	//gain for each feature value of each bag
//for finding the best split
float_point *BagManager::m_pfLocalBestGainEachBag_d = NULL;	//local best gain of each bag
int *BagManager::m_pnLocalBestGainKeyEachBag_d = NULL;		//local best gain key of each bag
int BagManager::m_maxNumofBlockPerNode = -1;				//number of blocks
float_point *BagManager::m_pfGlobalBestGainEachBag_d = NULL;//global best gain of each bag
int *BagManager::m_pnGlobalBestGainKeyEachBag_d = NULL;		//global best gain key of each bag
int *BagManager::m_pMaxNumValuePerFeaEachBag = NULL;
int *BagManager::m_pEachFeaLenEachNodeEachBag_dh = NULL;//each feature value length in each node

//for pinned memory; for computing indices in multiple level tree
unsigned int *BagManager::m_pIndicesEachBag_d = NULL;	//indices for multiple level tree of each bag
unsigned int *BagManager::m_pNumFvalueEachNodeEachBag_d = NULL;	//the number of feature values of each (splittable?) node
unsigned int *BagManager::m_pFvalueStartPosEachNodeEachBag_d = NULL;//the start position of each node
unsigned int *BagManager::m_pEachFeaStartPosEachNodeEachBag_d = NULL;//the start position of each feature in a node
int *BagManager::m_pEachFeaLenEachNodeEachBag_d = NULL;	//the number of values of each feature in each node

//memory for each individual tree
int *BagManager::m_pNumofNodeEachTreeEachBag = NULL;	//the number of nodes of each tree
int *BagManager::m_pStartPosOfEachTreeEachBag = NULL;	//the start position of each tree in the memory

//memory for splittable nodes
TreeNode *BagManager::m_pSplittableNodeEachBag = NULL;
SplitPoint *BagManager::m_pBestSplitPointEachBag = NULL;//(require memset!) store the best split points
nodeStat *BagManager::m_pSNodeStatEachBag = NULL;	//splittable node statistics
nodeStat *BagManager::m_pRChildStatEachBag = NULL;
nodeStat *BagManager::m_pLChildStatEachBag = NULL;
nodeStat *BagManager::m_pTempRChildStatEachBag = NULL;//(require memset!) store temporary statistics of right child
float_point *BagManager::m_pLastValueEachBag = NULL;//store the last processed value (for computing split point)
int *BagManager::m_nSNLockEachBag = NULL;
int *BagManager::m_curNumofSplitableEachBag_h = NULL; //number of splittable node of current tree
int *BagManager::m_pSNIdToBuffIdEachBag = NULL;	//(require memset!) map splittable node id to buffer position
int *BagManager::m_pPartitionId2SNPosEachBag = NULL;	//store all the buffer ids for splittable nodes
int *BagManager::m_pNumofBuffIdEachBag = NULL;	//the total number of buffer ids in the current round.
//host memory for GPU memory reset
SplitPoint *BagManager::m_pBestPointEachBagHost = NULL;//best split points

TreeNode *BagManager::m_pNodeTreeOnTrainingEachBag = NULL;//reserve memory for all nodes of the tree
//current numof nodes
int *BagManager::m_pCurNumofNodeTreeOnTrainingEachBag_d = NULL;
int *BagManager::m_pNumofNewNodeTreeOnTrainingEachBag = NULL;
//host memory for reset purposes
TreeNode *BagManager::m_pNodeTreeOnTrainingEachBagHost = NULL;

//memory for parent node to children ids
int *BagManager::m_pParentIdEachBag = NULL;
int *BagManager::m_pLeftChildIdEachBag = NULL;
int *BagManager::m_pRightChildIdEachBag = NULL;
//memory for new node statistics
nodeStat *BagManager::m_pNewNodeStatEachBag = NULL;
TreeNode *BagManager::m_pNewSplittableNodeEachBag = NULL;
//memory for used features in the current splittable nodes
int *BagManager::m_pFeaIdToBuffIdEachBag = NULL;//(require memset!) map feature id to buffer id
int *BagManager::m_pUniqueFeaIdVecEachBag = NULL;	//store all the used feature ids
int *BagManager::m_pNumofUniqueFeaIdEachBag = NULL;//(require memset!)store the number of unique feature ids
int BagManager::m_maxNumUsedFeaATree = -1;	//for reserving GPU memory; maximum number of used features in a tree

//temp host variable
float_point *BagManager::m_pTrueLabel_h = NULL;

int *BagManager::m_pPreMaxNid_h = NULL;

int *BagManager::m_pHashFeaIdToDenseInsPosBag = NULL;	//hash map for used feature ids of all trees to the dense instance position
int *BagManager::m_pSortedUsedFeaIdBag = NULL;			//sorted used feature ids

/**
 * @brief: initialise bag manager
 */
void BagManager::InitBagManager(int numIns, int numFea, int numTree, int numBag, int maxNumSN, int maxNumNode, long long numFeaValue,
								int maxNumUsedFeaInATree, int maxTreeDepth)
{
	int deviceId = -1;
	cudaGetDevice(&deviceId);
	printf("device id=%d\n", deviceId);

	GETERROR("error before init bag manager");
	PROCESS_ERROR(numIns > 0 && numBag > 0 && maxNumSN > 0 && maxNumNode > 0);
	m_numIns = numIns;
	m_numFea = numFea;
	m_numBag = numBag;
	m_numFeaValue = numFeaValue;

	//tree info
	m_numTreeEachBag = numTree;
	m_maxNumSplittable = maxNumSN;
	m_maxNumNode = maxNumNode;
	m_maxTreeDepth = maxTreeDepth;

	m_maxNumUsedFeaATree = maxNumUsedFeaInATree;

	BagBuilder bagBuilder;
	m_pInsWeight = new int[m_numIns * m_numBag];//bags are represented by weights to each instance
	bagBuilder.ContructBag(m_numIns, m_pInsWeight, m_numBag);
#if false
	for(int i = 0; i < m_numBag; i++)
	{
		int total = 0;
		for(int j = 0; j < m_numIns; j++)
		{
			total += m_pInsWeight[j + i * m_numIns];
		}
		if(total != m_numIns)
			cerr << "error in building bags" << endl;
	}
#endif
	GETERROR("error before create stream");

	printf("# of bags=%d\n", m_numBag);
	m_pStream = new cudaStream_t[numBag];
	for(int i = 0; i < m_numBag; i++)
		cudaStreamCreate(&m_pStream[i]);

	GETERROR("before allocate memory in BagManager");
	AllocMem();

	//initialise device memory
	GPUMemManager manager;
	cudaMemcpy(m_pInsWeight_d, m_pInsWeight, sizeof(int) * m_numIns * m_numBag, cudaMemcpyHostToDevice);
	cudaMemset(m_pInsIdToNodeIdEachBag, 0, sizeof(int) * m_numIns * m_numBag);
}

/**
 * @brief: allocate device memory for each bag
 */
void BagManager::AllocMem()
{
	//instance information for each bag
	PROCESS_ERROR(m_numIns > 0 && m_numBag > 0);
	checkCudaErrors(cudaMalloc((void**)&m_pInsIdToNodeIdEachBag, sizeof(int) * m_numIns * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pInsWeight_d, sizeof(int) * m_numIns * m_numBag));

	/******* memory for find split ******/
	//predicted value, gradient, hessian
	checkCudaErrors(cudaMalloc((void**)&m_pPredBufferEachBag, sizeof(float_point) * m_numIns * m_numBag));
	checkCudaErrors(cudaMemset(m_pPredBufferEachBag, 0, sizeof(float_point) * m_numIns * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pdDenseInsEachBag, sizeof(float_point) * m_maxNumUsedFeaATree * m_numIns * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pTargetValueEachBag, sizeof(float_point) * m_numIns * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pdTrueTargetValueEachBag, sizeof(float_point) * m_numIns * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pInsGradEachBag, sizeof(float_point) * m_numIns * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pInsHessEachBag, sizeof(float_point) * m_numIns * m_numBag));
	m_numBlockForBlockSum = ceil(m_numIns / 64.0);
	checkCudaErrors(cudaMalloc((void**)&m_pGDBlockSumEachBag, sizeof(float_point) * m_numBlockForBlockSum  * m_numBag));//64 is the min # of elements for a block sum
	checkCudaErrors(cudaMalloc((void**)&m_pHessBlockSumEachBag, sizeof(float_point) * m_numBlockForBlockSum  * m_numBag));

	//gradient and hessian prefix sum
	checkCudaErrors(cudaMalloc((void**)&m_pGDEachFvalueEachBag, sizeof(float_point) * m_numFeaValue * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pHessEachFvalueEachBag, sizeof(float_point) * m_numFeaValue * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pDenseFValueEachBag, sizeof(float_point) * m_numFeaValue * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pGDPrefixSumEachBag, sizeof(float_point) * m_numFeaValue * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pHessPrefixSumEachBag, sizeof(float_point) * m_numFeaValue * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pGainEachFvalueEachBag, sizeof(float_point) * m_numFeaValue * m_numBag));
	//for finding the best split
	int blockSizeLocalBest;
	dim3 tempNumofBlockLocalBest;
	KernelConf conf;
	conf.ConfKernel(m_numFeaValue, blockSizeLocalBest, tempNumofBlockLocalBest);
	m_maxNumofBlockPerNode = tempNumofBlockLocalBest.x * tempNumofBlockLocalBest.y;
	checkCudaErrors(cudaMalloc((void**)&m_pfLocalBestGainEachBag_d, sizeof(float_point) * m_maxNumofBlockPerNode * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pnLocalBestGainKeyEachBag_d, sizeof(int) * m_maxNumofBlockPerNode * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pfGlobalBestGainEachBag_d, sizeof(float_point) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pnGlobalBestGainKeyEachBag_d, sizeof(int) * m_maxNumSplittable * m_numBag));

	m_pMaxNumValuePerFeaEachBag = new int[m_numBag];
	checkCudaErrors(cudaMallocHost((void**)&m_pEachFeaLenEachNodeEachBag_dh, sizeof(int) * m_maxNumSplittable * m_numFea * m_numBag));
	//initialise length of each feature in each node
	memset(m_pEachFeaLenEachNodeEachBag_dh, 0, sizeof(int) * m_numFea * m_maxNumSplittable * m_numBag);

	//corresponding to pinned memory; for computing indices of more than one level trees
	checkCudaErrors(cudaMalloc((void**)&m_pIndicesEachBag_d, sizeof(unsigned int) * m_numFeaValue * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pNumFvalueEachNodeEachBag_d, sizeof(unsigned int) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pFvalueStartPosEachNodeEachBag_d, sizeof(unsigned int) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pEachFeaStartPosEachNodeEachBag_d, sizeof(unsigned int) * m_maxNumSplittable * m_numBag * m_numFea));
	checkCudaErrors(cudaMalloc((void**)&m_pEachFeaLenEachNodeEachBag_d, sizeof(int) * m_maxNumSplittable * m_numBag * m_numFea));

	/********** memory for splitting node ************/
	//for individual tree
	checkCudaErrors(cudaMalloc((void**)&m_pNumofNodeEachTreeEachBag, sizeof(int) * m_numTreeEachBag * m_numBag));
	checkCudaErrors(cudaMemset(m_pNumofNodeEachTreeEachBag, 0, sizeof(int) * m_numTreeEachBag * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pStartPosOfEachTreeEachBag, sizeof(int) * m_numTreeEachBag * m_numBag));
	checkCudaErrors(cudaMemset(m_pStartPosOfEachTreeEachBag, -1, sizeof(int) * m_numTreeEachBag * m_numBag));

	//for splittable nodes
	checkCudaErrors(cudaMalloc((void**)&m_pSplittableNodeEachBag, sizeof(TreeNode) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pBestSplitPointEachBag, sizeof(SplitPoint) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pSNodeStatEachBag, sizeof(nodeStat) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pRChildStatEachBag, sizeof(nodeStat) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pLChildStatEachBag, sizeof(nodeStat) * m_maxNumSplittable * m_numBag));
	//temporary space for splittable nodes
	checkCudaErrors(cudaMalloc((void**)&m_pTempRChildStatEachBag, sizeof(nodeStat) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pLastValueEachBag, sizeof(float_point) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMemset(m_pLastValueEachBag, 0, sizeof(float_point) * m_maxNumSplittable * m_numBag));
	m_curNumofSplitableEachBag_h = new int[m_numBag];
	checkCudaErrors(cudaMalloc((void**)&m_nSNLockEachBag, sizeof(int) * m_numBag));//a lock for critical region
	checkCudaErrors(cudaMemset(m_nSNLockEachBag, 0, sizeof(int) * m_numBag));
	//map splittable node to buffer id
	checkCudaErrors(cudaMalloc((void**)&m_pSNIdToBuffIdEachBag, sizeof(int) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pPartitionId2SNPosEachBag, sizeof(int) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pNumofBuffIdEachBag, sizeof(int) * m_numBag));
	checkCudaErrors(cudaMemset(m_pPartitionId2SNPosEachBag, -1, sizeof(int) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMemset(m_pNumofBuffIdEachBag, 0, sizeof(int) * m_numBag));
	m_pBestPointEachBagHost = new SplitPoint[m_maxNumSplittable * m_numBag];
	//for parent and children relationship
	checkCudaErrors(cudaMalloc((void**)&m_pParentIdEachBag, sizeof(int) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pLeftChildIdEachBag, sizeof(int) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pRightChildIdEachBag, sizeof(int) * m_maxNumSplittable * m_numBag));
	//memory for new node statistics
	checkCudaErrors(cudaMalloc((void**)&m_pNewNodeStatEachBag, sizeof(nodeStat) * m_maxNumSplittable * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pNewSplittableNodeEachBag, sizeof(TreeNode) * m_maxNumSplittable * m_numBag));
	//map splittable node to buffer id
	checkCudaErrors(cudaMalloc((void**)&m_pFeaIdToBuffIdEachBag, sizeof(int) * m_maxNumUsedFeaATree * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pUniqueFeaIdVecEachBag, sizeof(int) * m_maxNumUsedFeaATree * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pNumofUniqueFeaIdEachBag, sizeof(int) * m_numBag));

	/*********** memory for the tree (on training and final) ******/
	checkCudaErrors(cudaMalloc((void**)&m_pNodeTreeOnTrainingEachBag, sizeof(TreeNode) * m_maxNumNode * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pCurNumofNodeTreeOnTrainingEachBag_d, sizeof(int) * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pNumofNewNodeTreeOnTrainingEachBag, sizeof(int) * m_numBag));
	//for reseting memory for the next tree
	m_pNodeTreeOnTrainingEachBagHost = new TreeNode[m_maxNumNode * m_numBag];
	//for final trees
	checkCudaErrors(cudaMalloc((void**)&m_pAllTreeEachBag, sizeof(TreeNode) * m_numTreeEachBag * m_maxNumNode * m_numBag));
	//memory set for all tree nodes
	TreeNode *pAllTreeNodeHost = new TreeNode[m_numTreeEachBag * m_maxNumNode * m_numBag];
	cudaMemcpy(m_pAllTreeEachBag, pAllTreeNodeHost, sizeof(TreeNode) * m_numTreeEachBag * m_maxNumNode * m_numBag, cudaMemcpyHostToDevice);
	delete[] pAllTreeNodeHost;

	m_pNumofTreeLearntEachBag_h = new int[m_numBag];
	memset(m_pNumofTreeLearntEachBag_h, 0, sizeof(int) * m_numBag);

	//for the currently constructed tree
	checkCudaErrors(cudaMalloc((void**)&m_pHashFeaIdToDenseInsPosBag, sizeof(int) * m_maxNumUsedFeaATree * m_numBag));
	checkCudaErrors(cudaMemset(m_pHashFeaIdToDenseInsPosBag, -1, sizeof(int) * m_maxNumUsedFeaATree * m_numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pSortedUsedFeaIdBag, sizeof(int) * m_maxNumUsedFeaATree * m_numBag));

	/***** memory for others******/
	checkCudaErrors(cudaMalloc((void**)&m_pSNBufferEachBag, sizeof(TreeNode) * m_maxNumSplittable * m_numBag));

	m_pPreMaxNid_h = new int[m_numBag];
}

void BagManager::FreeMem()
{
	cudaFree(m_pSNBufferEachBag);
	cudaFree(m_pInsIdToNodeIdEachBag);
	cudaFree(m_pInsWeight_d);
	cudaFree(m_pAllTreeEachBag);
	cudaFree(m_pTargetValueEachBag);
	cudaFree(m_pInsGradEachBag);
	cudaFree(m_pInsHessEachBag);
	delete []m_pInsWeight;
}
