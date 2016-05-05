/*
 * gbdtMemManager.h
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef GBDTMEMMANAGER_H_
#define GBDTMEMMANAGER_H_

#include "gpuMemManager.h"

class GBDTGPUMemManager: public GPUMemManager
{
public:
	static int *pDInsId;
	static double *pdDFeaValue;
	static int *pDNumofKeyValue;

	static long long totalNumofValues;
	static int m_numofFea;

public:
	void allocMemForIns(int nTotalNumofValue, int numofFeature);
};



#endif /* GBDTMEMMANAGER_H_ */
