/*
 * BagOrgManager.cu
 *
 *  Created on: Jul 24, 2017
 *      Author: zeyi
 */

#include <helper_cuda.h>

#include "BagOrgManager.h"

real *BagOrgManager::m_pDenseFValueEachBag = NULL;		//feature values of consideration (use for computing the split?)
double *BagOrgManager::m_pdGDPrefixSumEachBag = NULL;		//gd prefix sum for each feature
real *BagOrgManager::m_pHessPrefixSumEachBag = NULL;	//hessian prefix sum for each feature
real *BagOrgManager::m_pGainEachFvalueEachBag = NULL;	//gain for each feature value of each bag

BagOrgManager::BagOrgManager(uint numFeaValue, int numBag){
	if(m_pDenseFValueEachBag != NULL)
		return;
	//gradient and hessian prefix sum
	checkCudaErrors(cudaMalloc((void**)&m_pDenseFValueEachBag, sizeof(real) * numFeaValue * numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pdGDPrefixSumEachBag, sizeof(double) * numFeaValue * numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pHessPrefixSumEachBag, sizeof(real) * numFeaValue * numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pGainEachFvalueEachBag, sizeof(real) * numFeaValue * numBag));
}
