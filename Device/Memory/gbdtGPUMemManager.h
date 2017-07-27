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
	//memory for instances (key on feature id); for training
	static int *m_pDInsId, *m_pDNumofKeyValue;
	static uint *m_pFeaStartPos;

	//memory for instances (key on instance id); for prediction
	static int *m_pDFeaId, *m_pDNumofFea;
	static real *m_pdDInsValue;
	static uint *m_pInsStartPos;
	//with bag info
	static real *m_pdDenseInsEachBag;
	static int *m_pHashFeaIdToDenseInsPosBag;
	static int *m_pSortedUsedFeaIdBag;
	static TreeNode *m_pAllTreeEachBag;
	static int *m_pNumofTreeLearntEachBag_h;
	static int *m_pStartPosOfEachTreeEachBag;
	static int *m_pNumofNodeEachTreeEachBag;

	//memory for prediction
	static int m_maxUsedFeaInATree;

	static uint m_numFeaValue;
	static int m_numofIns, m_numofFea;

public:
	void mallocForTrainingIns(int nTotalNumofValue, int numofIns, int numofFeature);
	void freeMemForTrainingIns();

	void mallocForTestingIns(int nTotalNumofValue, int numofIns, int numofFeature,
							 int numBag, int numTreeABag, int maxNumNode);
	void freeMemForTestingIns();
};

#endif /* GBDTMEMMANAGER_H_ */
