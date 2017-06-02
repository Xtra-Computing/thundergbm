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
	static int *m_pFvalueFid_d;
	static real *m_pdDFeaValue;
	static unsigned int *m_pFeaStartPos;
	//memory for instances (key on instance id)
	static int *m_pDFeaId, *m_pDNumofFea;
	static real *m_pdDInsValue;
	static uint *m_pInsStartPos;

	//memory for prediction
	static int m_maxUsedFeaInTrees;

	static unsigned int m_numFeaValue;
	static int m_numofIns, m_numofFea;

public:
	void allocMemForIns(int nTotalNumofValue, int numofIns, int numofFeature);
	void freeMemForIns();
};

#endif /* GBDTMEMMANAGER_H_ */
