/*
 * BagManager.h
 *
 *  Created on: 8 Aug 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef BAGMANAGER_H_
#define BAGMANAGER_H_

#include <helper_cuda.h>
#include <driver_types.h>
#include <cuda.h>

#include "../../DeviceHost/TreeNode.h"
#include "../../SharedUtility/DataType.h"
#include "../../DeviceHost/NodeStat.h"
#include "../../Host/UpdateOps/SplitPoint.h"

class BagManager
{
public:
	static int *m_pInsWeight;
	static int m_numBag;
	static int m_numIns;
	static int m_numFea;
	static uint m_numFeaValue;

	//tree info
	static int m_numTreeEachBag;
	static int m_maxNumNode;
	static int m_maxNumSplittable;
	static int m_maxTreeDepth;
	static int m_maxNumLeave;

	//device memory
	static cudaStream_t *m_pStream;
	static int *m_pInsIdToNodeIdEachBag;
	static int *m_pInsWeight_d;

	//for gradient and hessian
	static real *m_pTargetValueEachBag;
	static real *m_pdTrueTargetValueEachBag;
	static real *m_pInsGradEachBag, *m_pInsHessEachBag;

	static real *m_pDenseFValueEachBag;

	//for pinned memory; for computing indices in multiple level tree
	static uint *m_pIndicesEachBag_d;
	static uint *m_pNumFvalueEachNodeEachBag_d;//the number of feature values of each (splittable?) node
	static uint *m_pFvalueStartPosEachNodeEachBag_d;//the start position of each node
	static uint *m_pEachFeaStartPosEachNodeEachBag_d;//the start position of each feature in a node
	static int *m_pEachFeaLenEachNodeEachBag_d;//the number of values of each feature in each node

	//for splittable nodes
	static TreeNode *m_pSplittableNodeEachBag;
	static SplitPoint *m_pBestSplitPointEachBag;
	static nodeStat *m_pSNodeStatEachBag;
	static nodeStat *m_pRChildStatEachBag;
	static nodeStat *m_pLChildStatEachBag;
	//temporary space for splittable nodes
	static int *m_curNumofSplitableEachBag_h;
	//map splittable node to buffer id
	static int *m_pPartitionId2SNPosEachBag;

	//memory for the tree on training
	static TreeNode *m_pNodeTreeOnTrainingEachBag;
	static int *m_pCurNumofNodeTreeOnTrainingEachBag_d;
	static int *m_pNumofNewNodeTreeOnTrainingEachBag;
	//memory for new node statistics
	static nodeStat *m_pNewNodeStatEachBag;
	static TreeNode *m_pNewNodeEachBag;

	//memory for parent node to children ids
	static int *m_pParentIdEachBag;
	static int *m_pLeftChildIdEachBag, *m_pRightChildIdEachBag;
	//memory for used features in the current splittable nodes
	static int *m_pFeaIdToBuffIdEachBag;
	static int *m_pUniqueFeaIdVecEachBag;
	static int *m_pNumofUniqueFeaIdEachBag;
	static int m_maxNumUsedFeaATree;

	//temp memory
	static real *m_pTrueLabel_h;

	//preMaxNid
	static int *m_pPreMaxNid_h;
	static uint *m_pPreNumSN_h;

public:
	static void InitBagManager(int numIns, int numFea, int numTree, int numBag, int maxNumSN,
							   int maxNumNode, long long numFeaValue, int maxNumUsedFeaInATree, int maxTreeDepth);

	static void FreeMem();
private:
	static void AllocMem();
};



#endif /* BAGMANAGER_H_ */
