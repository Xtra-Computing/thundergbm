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
#include "../../DeviceHost/TreeNode.h"
#include "../../DeviceHost/NodeStat.h"
#include "../../Host/UpdateOps/SplitPoint.h"

class GBDTGPUMemManager: public GPUMemManager
{
public:
	//memory for instances (key on feature id)
	static int *m_pDInsId, *m_pDNumofKeyValue;
	static float_point *m_pdDFeaValue;
	static long long *m_pFeaStartPos;
	//memory for instances (key on instance id)
	static int *m_pDFeaId, *m_pDNumofFea;
	static float_point *m_pdDInsValue;
	static long long *m_pInsStartPos;

	//memory for prediction
	static float_point *m_pPredBuffer;
	static float_point *m_pdTrueTargetValue;
	static float_point *m_pdDenseIns;
	static float_point *m_pTargetValue;
	static int maxNumofDenseIns;
	static int *m_pHashFeaIdToDenseInsPos;
	static int *m_pSortedUsedFeaId;
	static int m_maxUsedFeaInTrees;

	static int *m_pInsIdToNodeId;
	static long long m_totalNumofValues;
	static int m_numofIns, m_numofFea;

	//memory for gradient and hessian
	static float_point *m_pGrad, *m_pHess;

	//memory for splittable nodes
	static int m_maxNumofSplittable;
	static int m_curNumofSplitable;
	static TreeNode *m_pSplittableNode;
	static SplitPoint *m_pBestSplitPoint;
	static nodeStat *m_pSNodeStat, *m_pRChildStat, *m_pLChildStat, *m_pTempRChildStat;
	static float_point *m_pLastValue;
	static int *m_nSNLock;

	//memory for finding best split for each feature on each node
	static nodeStat *m_pRChildStatPerThread, *m_pLChildStatPerThread, *m_pTempRChildStatPerThread;
	static float_point *m_pLastValuePerThread;
	static SplitPoint *m_pBestSplitPointPerThread;

	//map splittable node id to buffer position
	static int *m_pSNIdToBuffId, *m_pBuffIdVec, *m_pNumofBuffId;

	//model param
	static float_point m_lambda;

	//for host memory (use to reset GPU memory)
	static SplitPoint *m_pBestPointHost;
	static SplitPoint *m_pBestPointHostPerThread;

public:
	void allocMemForIns(int nTotalNumofValue, int numofIns, int numofFeature);
	void allocMemForSplittableNode(int nMaxNumofSplittableNode);
	void allocMemForSNForEachThread(int maxNumofThread, int maxNumofSplittable);
	void freeMemForSNForEachThread();

	void allocHostMemory();
	void releaseHostMemory();
};



#endif /* GBDTMEMMANAGER_H_ */
