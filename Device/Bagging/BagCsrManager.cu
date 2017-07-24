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
int *BagCsrManager::preFvalueInsId = NULL;
uint BagCsrManager::curNumCsr = 0;
uint BagCsrManager::reservedMaxNumCsr = pow(2, 20);
uint *BagCsrManager::pCsrLen = NULL;
double *BagCsrManager::pCsrGD = NULL;
real *BagCsrManager::pCsrHess = NULL;
real *BagCsrManager::pCsrFvalue = NULL;
bool *BagCsrManager::pCsrDefault2Right = NULL;
real *BagCsrManager::pCsrGain = NULL;
uint *BagCsrManager::pCsrKey = NULL;

BagCsrManager::BagCsrManager(int numFea, int maxNumSN, uint totalNumFeaValue){
	if(pCsrLen != NULL)//already reserved memory
		return;

	curNumCsr = 0;
	reservedMaxNumCsr = pow(2, 20);//1M
	checkCudaErrors(cudaMalloc((void**)&pCsrLen, sizeof(uint) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrGD, sizeof(double) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrHess, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrFvalue, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrDefault2Right, sizeof(bool) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrGain, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrKey, sizeof(uint) * reservedMaxNumCsr));

	checkCudaErrors(cudaMalloc((void**)&pEachCsrFeaStartPos, sizeof(uint) * numFea * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrFeaLen, sizeof(uint) * numFea * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrNodeStartPos, sizeof(uint) * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachNodeSizeInCsr, sizeof(uint) * maxNumSN));

	checkCudaErrors(cudaMalloc((void**)&preFvalueInsId, sizeof(int) * totalNumFeaValue));
}

void BagCsrManager::reserveSpace(){
	checkCudaErrors(cudaFree(pCsrLen));
	checkCudaErrors(cudaFree(pCsrGD));
	checkCudaErrors(cudaFree(pCsrHess));
	checkCudaErrors(cudaFree(pCsrFvalue));
	checkCudaErrors(cudaFree(pCsrDefault2Right));
	checkCudaErrors(cudaFree(pCsrGain));
	checkCudaErrors(cudaFree(pCsrKey));
	//reserve larger memory
	checkCudaErrors(cudaMalloc((void**)&pCsrLen, sizeof(uint) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrGD, sizeof(double) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrHess, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**) &pCsrFvalue, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrDefault2Right, sizeof(bool) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrGain, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrKey, sizeof(uint) * reservedMaxNumCsr));
}

uint *BagCsrManager::getMutableCsrLen(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrLen != NULL);
	return pCsrLen;
}
double *BagCsrManager::getMutableCsrGD(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrGD != NULL);
	return pCsrGD;
}
real *BagCsrManager::getMutableCsrHess(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrHess != NULL);
	return pCsrHess;
}
real *BagCsrManager::getMutableCsrFvalue(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrFvalue != NULL);
	return pCsrFvalue;
}
real *BagCsrManager::getMutableCsrGain(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrGain != NULL);
	return pCsrGain;
}
uint *BagCsrManager::getMutableCsrKey(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrKey != NULL);
	return pCsrKey;
}
bool *BagCsrManager::getMutableDefault2Right(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveSpace();
	}
	PROCESS_ERROR(pCsrDefault2Right != NULL);
	return pCsrDefault2Right;
}

const uint *BagCsrManager::getCsrLen(){
	PROCESS_ERROR(pCsrLen != NULL);
	return pCsrLen;
}
const double *BagCsrManager::getCsrGD(){
	PROCESS_ERROR(pCsrGD != NULL);
	return pCsrGD;
}
const real *BagCsrManager::getCsrHess(){
	PROCESS_ERROR(pCsrHess != NULL);
	return pCsrHess;
}
const real *BagCsrManager::getCsrFvalue(){
	PROCESS_ERROR(pCsrFvalue != NULL);
	return pCsrFvalue;
}
const real *BagCsrManager::getCsrGain(){
	PROCESS_ERROR(pCsrGain != NULL);
	return pCsrGain;
}
const uint *BagCsrManager::getCsrKey(){
	PROCESS_ERROR(pCsrKey != NULL);
	return pCsrKey;
}
const bool *BagCsrManager::getDefault2Right(){
	PROCESS_ERROR(pCsrDefault2Right != NULL);
	return pCsrDefault2Right;
}
