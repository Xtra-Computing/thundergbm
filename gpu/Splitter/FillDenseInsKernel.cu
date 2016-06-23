/*
 * FillDenseInsKernel.cu
 *
 *  Created on: 21 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "DevicePredKernel.h"
#include "../DeviceHashing.h"
#include "../ErrorChecker.h"

__global__ void FillDense(float_point *pdSparseInsValue, int *pnSpareInsFeaId, int numofFeaValue,
						  float_point *pdDenseIns, int *pSortedUsedFea, int *pHashFeaIdToDenseInsPos, int totalUsedFea)
{
	//for each value in the sparse instance
	ErrorChecker(numofFeaValue - 1, __PRETTY_FUNCTION__, "numofFeaValue <= 0");
	int curDenseTop = 0;
	for(int i = 0; i < numofFeaValue; i++)
	{
		int feaId = pnSpareInsFeaId[i];

		while(feaId > pSortedUsedFea[curDenseTop])
		{
			int pos = GetBufferId(pHashFeaIdToDenseInsPos, pSortedUsedFea[curDenseTop], totalUsedFea);
			pdDenseIns[pos] = 0;
			curDenseTop++;
		}

		if(feaId == pSortedUsedFea[curDenseTop])
		{//this is a feature needed to be stored in dense instance
			int pos = GetBufferId(pHashFeaIdToDenseInsPos, pSortedUsedFea[curDenseTop], totalUsedFea);
			pdDenseIns[pos] = pdSparseInsValue[i];
			curDenseTop++;
		}
	}

}


