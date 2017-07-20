/*
 * ComputeGainDense.cu
 *
 *  Created on: 22 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: kernels gain computing using dense arrays
 */

#include <stdio.h>
#include <float.h>
#include <limits>
#include "FindFeaKernel.h"
#include "../Splitter/DeviceSplitter.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/binarySearch.h"

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void LoadGDHessFvalueRoot(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue)
{
	//one thread loads one value
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	int insId = pInsId[gTid];//instance id

	CONCHECKER(insId < numIns);

	//store GD and Hess.
	pGDEachFeaValue[gTid] = pInsGD[insId];
	pHessEachFeaValue[gTid] = pInsHess[insId];
	pDenseFeaValue[gTid] = pAllFeaValue[gTid];
}

/**
 * @brief: copy the gd, hess and feaValue for each node based on some features on similar number of values
 */
__global__ void LoadGDHessFvalue(const real *pInsGD, const real *pInsHess, int numIns,
						   const int *pInsId, const real *pAllFeaValue, const unsigned int *pDstIndexEachFeaValue, int numFeaValue,
						   double *pGDEachFeaValue, real *pHessEachFeaValue, real *pDenseFeaValue)
{
	//one thread loads one value
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaValue)//thread has nothing to load
		return;

	int insId = pInsId[gTid];//instance id

	CONCHECKER(insId < numIns);

	//index for scatter
	uint idx = pDstIndexEachFeaValue[gTid];
	if(idx == LARGE_4B_UINT)//instance is in a leaf node
		return;

	CONCHECKER(idx < numFeaValue);

	//scatter: store GD, Hess and the feature value.
	pGDEachFeaValue[idx] = pInsGD[insId];
	pHessEachFeaValue[idx] = pInsHess[insId];
	pDenseFeaValue[idx] = pAllFeaValue[gTid];
}

/**
 * @brief: change the gain of the first value of each feature to 0
 */
__global__ void FirstFeaGain(const unsigned int *pEachFeaStartPosEachNode, int numFeaStartPos, real *pGainOnEachFeaValue, uint numFeaValue)
{
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaStartPos)//no gain to fix
		return;
	unsigned int gainPos = pEachFeaStartPosEachNode[gTid];
	if(gainPos >= numFeaValue)
		return;//there may be some ending 0s (e.g. the last node has some features with any values).
	pGainOnEachFeaValue[gainPos] = 0;
}
