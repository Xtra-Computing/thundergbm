/*
 * gbdtMemManager.h
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef GBDTMEMMANAGER_H_
#define GBDTMEMMANAGER_H_

#include <helper_cuda.h>
#include "gpuMemManager.h"
#include "../../pureHost/Tree/TreeNode.h"
#include "../../pureHost/UpdateOps/SplitPoint.h"
#include "../../pureHost/UpdateOps/NodeStat.h"

class GBDTGPUMemManager: public GPUMemManager
{
public:
	//memory for instances
	static int *m_pDInsId, *m_pDNumofKeyValue;
	static float_point *m_pdDFeaValue;
	static long long *m_pFeaStartPos;

	static int *m_pInsIdToNodeId;
	static long long m_totalNumofValues;
	static int m_numofIns, m_numofFea;

	//memory for gradient and hessian
	static float_point *m_pGrad, *m_pHess;

	//memory for splittable nodes
	static int m_maxNumofSplittable;
	static TreeNode *m_pSplittableNode;
	static SplitPoint *m_pBestSplitPoint;
	static nodeStat *m_pSNodeStat, *m_pRChildStat, *m_pLChildStat, *m_pTempRChildStat;
	static float_point *m_pLastValue;

	//map splittable node id to buffer position
	static int *m_pSNIdToBuffId, *m_pBuffIdVec;

	//model param
	static float_point m_lambda;

	//for used features in a tree
	static int m_maxNumofUsedFea;
	static int *m_pFeaIdToBuffId;
	static int *m_pUniqueFeaIdVec;

	//for host memory (use to reset GPU memory)
	static SplitPoint *m_pBestPointHost;
	static int *m_pInsToNodeId;

public:
	void allocMemForIns(int nTotalNumofValue, int numofIns, int numofFeature);
	void allocMemForSplittableNode(int nMaxNumofSplittableNode);
	void allocMemForSplitting(int nMaxNumofUsedFeature);

	void allocHostMemory();
	void releaseHostMemory();
};



#endif /* GBDTMEMMANAGER_H_ */
