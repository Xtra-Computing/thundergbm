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
	int *pDInsId;
	double *pdDFeaValue;
	int *pDNumofKeyValue;

	long long totalNumofValues;

public:
	GBDTGPUMemManager();
	void allocMemForIns(int nTotalNumofValue, int numofFeature);
};



#endif /* GBDTMEMMANAGER_H_ */
