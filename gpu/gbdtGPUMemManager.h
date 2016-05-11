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
#include "../pureHost/Tree/TreeNode.h"
#include "../pureHost/UpdateOps/SplitPoint.h"
#include "../pureHost/UpdateOps/NodeStat.h"

class GBDTGPUMemManager: public GPUMemManager
{
public:
	//memory for instances
	static int *pDInsId;
	static float_point *pdDFeaValue;
	static int *pDNumofKeyValue;
	static long long *pFeaStartPos;

	static int *pInsIdToNodeId;
	static long long totalNumofValues;
	static int m_numofIns;
	static int m_numofFea;

	//memory for gradient and hessian
	static float_point *pGrad;
	static float_point *pHess;

	//memory for splittable nodes
	static int m_maxNumofSplittable;
	static TreeNode *pSplittableNode;
	static SplitPoint *pBestSplitPoint;
	static nodeStat *pSNodeStat;
	static nodeStat *pRChildStat;
	static nodeStat *pLChildStat;

	static nodeStat *pTempRChildStat;
	static float_point *pLastValue;

	//map splittable node id to buffer position
	static int *pSNIdToBuffId;
	static int *pBuffIdVec;

	//model param
	static float_point m_lambda;

public:
	void allocMemForIns(int nTotalNumofValue, int numofIns, int numofFeature);
	void allocMemForSplittableNode(int nMaxNumofSplittableNode);
};



#endif /* GBDTMEMMANAGER_H_ */
