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
real *BagOrgManager::m_pdDFeaValue = NULL;
uint *BagOrgManager::m_pnKey_d = NULL;
uint *BagOrgManager::m_pnTid2Fid = NULL;
bool BagOrgManager::needCopy = true;

BagOrgManager::BagOrgManager(uint numFeaValue, int numBag){
	if(m_pDenseFValueEachBag != NULL)
		return;
	checkCudaErrors(cudaMalloc((void**)&m_pdDFeaValue, sizeof(real) * numFeaValue * numBag));

	checkCudaErrors(cudaMalloc((void**)&m_pnKey_d, numFeaValue * sizeof(uint) * numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pnTid2Fid, numFeaValue * sizeof(uint) * numBag));
	//gradient and hessian prefix sum
	checkCudaErrors(cudaMalloc((void**)&m_pDenseFValueEachBag, sizeof(real) * numFeaValue * numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pdGDPrefixSumEachBag, sizeof(double) * numFeaValue * numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pHessPrefixSumEachBag, sizeof(real) * numFeaValue * numBag));
	checkCudaErrors(cudaMalloc((void**)&m_pGainEachFvalueEachBag, sizeof(real) * numFeaValue * numBag));

}
