/*
 * BagCsrManager.cu
 *
 *  Created on: Jul 23, 2017
 *      Author: zeyi
 */

#include <helper_cuda.h>
#include "BagCsrManager.h"
#include "../../SharedUtility/CudaMacro.h"

uint *BagCsrManager::pEachCsrFeaStartPos = NULL;
uint *BagCsrManager::pEachCsrFeaLen = NULL;
uint *BagCsrManager::pEachCsrNodeStartPos = NULL;
uint *BagCsrManager::pEachNodeSizeInCsr = NULL;
uint BagCsrManager::curNumCsr = 0;
uint BagCsrManager::reservedMaxNumCsr = pow(2, 20);
uint *BagCsrManager::pCsrLen = NULL;
double *BagCsrManager::pCsrGD = NULL;
real *BagCsrManager::pCsrHess = NULL;
real *BagCsrManager::pCsrFvalue = NULL;

BagCsrManager::BagCsrManager(){
	if(pCsrLen != NULL)//already reserved memory
		return;

	curNumCsr = 0;
	reservedMaxNumCsr = pow(2, 20);//1M
	checkCudaErrors(cudaMalloc((void**)&pCsrLen, sizeof(uint) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrGD, sizeof(double) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrHess, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**) &pCsrFvalue, sizeof(real) * reservedMaxNumCsr));
}

void BagCsrManager::reserveSpace(){
	checkCudaErrors(cudaFree(pCsrLen));
	checkCudaErrors(cudaFree(pCsrGD));
	checkCudaErrors(cudaFree(pCsrHess));
	checkCudaErrors(cudaFree(pCsrFvalue));
	//reserve larger memory
	checkCudaErrors(cudaMalloc((void**)&pCsrLen, sizeof(uint) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrGD, sizeof(double) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrHess, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**) &pCsrFvalue, sizeof(real) * reservedMaxNumCsr));
}

uint *BagCsrManager::getMutableCsrLen(uint numCsr){
	curNumCsr = numCsr;
	if(reservedMaxNumCsr < numCsr){
		reservedMaxNumCsr = numCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrLen);
	return pCsrLen;
}
double *BagCsrManager::getMutableCsrGD(uint numCsr){
	curNumCsr = numCsr;
	if(reservedMaxNumCsr < numCsr){
		reservedMaxNumCsr = numCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrGD);
	return pCsrGD;
}
real *BagCsrManager::getMutableCsrHess(uint numCsr){
	curNumCsr = numCsr;
	if(reservedMaxNumCsr < numCsr){
		reservedMaxNumCsr = numCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrHess);
	return pCsrHess;
}
real *BagCsrManager::getMutableCsrFvalue(uint numCsr){
	curNumCsr = numCsr;
	if(reservedMaxNumCsr < numCsr){
		reservedMaxNumCsr = numCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrFvalue);
	return pCsrFvalue;
}

const uint *BagCsrManager::getCsrLen(){
	PROCESS_ERROR(pCsrLen);
	return pCsrLen;
}
const double *BagCsrManager::getCsrGD(){
	PROCESS_ERROR(pCsrGD);
	return pCsrGD;
}
const real *BagCsrManager::getCsrHess(){
	PROCESS_ERROR(pCsrHess);
	return pCsrHess;
}
const real *BagCsrManager::getCsrFvalue(){
	PROCESS_ERROR(pCsrFvalue);
	return pCsrFvalue;
}
