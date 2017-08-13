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
MemVector BagCsrManager::csrLen;//shared with pCsrStart
MemVector BagCsrManager::csrKey; //shared
real *BagCsrManager::pCsrFvalue = NULL;

BagCsrManager::BagCsrManager(int numFea, int maxNumSN, uint totalNumFeaValue){
	if(pCsrFvalue != NULL)//already reserved memory
		return;

	curNumCsr = 0;
	reservedMaxNumCsr = totalNumFeaValue/20;//10 times compression ratio

	checkCudaErrors(cudaMalloc((void**)&pCsrFvalue, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrFeaStartPos, sizeof(uint) * numFea * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrFeaLen, sizeof(uint) * numFea * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrNodeStartPos, sizeof(uint) * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachNodeSizeInCsr, sizeof(uint) * maxNumSN));

	checkCudaErrors(cudaMalloc((void**)&preFvalueInsId, sizeof(int) * totalNumFeaValue));
}

void BagCsrManager::reserveCsrSpace(){
	checkCudaErrors(cudaFree(pCsrFvalue));
	//reserve larger memory
	printf("max num of csr is %u\n", reservedMaxNumCsr);
	checkCudaErrors(cudaMalloc((void**) &pCsrFvalue, sizeof(real) * reservedMaxNumCsr));
}

/** operations on cross variable reused memory **/
uint *BagCsrManager::getMutableCsrLen(){
	PROCESS_ERROR(curNumCsr > 0);
	if(csrLen.reservedSize < curNumCsr)
		csrLen.reserveSpace(curNumCsr, sizeof(uint));
	PROCESS_ERROR(csrLen.addr != NULL);
	return (uint*)csrLen.addr;
}

uint *BagCsrManager::getMutableCsrKey(){
	PROCESS_ERROR(curNumCsr > 0);
	if(csrKey.reservedSize < curNumCsr)
		csrKey.reserveSpace(curNumCsr, sizeof(uint));
	PROCESS_ERROR(csrKey.addr != NULL);
	return (uint*)csrKey.addr;
}

uint *BagCsrManager::getMutableCsrStart(){
	return getMutableCsrLen();
}

const uint *BagCsrManager::getCsrLen(){
	PROCESS_ERROR(csrLen.addr != NULL);
	return (uint*)csrLen.addr;
}

const uint *BagCsrManager::getCsrKey(){
	PROCESS_ERROR(csrKey.addr != NULL);
	return (uint*)csrKey.addr;
}

const uint *BagCsrManager::getCsrStart(){
	return getCsrLen();
}

/* operations on not cross variable reused memory */
real *BagCsrManager::getMutableCsrFvalue(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveCsrSpace();
	}
	PROCESS_ERROR(pCsrFvalue != NULL);
	return pCsrFvalue;
}
const real *BagCsrManager::getCsrFvalue(){
	PROCESS_ERROR(pCsrFvalue != NULL);
	return pCsrFvalue;
}
