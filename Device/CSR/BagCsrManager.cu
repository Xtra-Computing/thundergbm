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
MemVector BagCsrManager::csrGD; //shared with pNewCsrLen
MemVector BagCsrManager::csrHess; //shared with pCsrFvalueSparse
MemVector BagCsrManager::csrGain; //shared with csrMarker
MemVector BagCsrManager::pCsrKey; //shared with pCsrStartCurRound
real *BagCsrManager::pCsrFvalue = NULL;
bool *BagCsrManager::pCsrDefault2Right = NULL;

BagCsrManager::BagCsrManager(int numFea, int maxNumSN, uint totalNumFeaValue){
	if(pCsrFvalue != NULL)//already reserved memory
		return;

	curNumCsr = 0;
	reservedMaxNumCsr = totalNumFeaValue/40;//40 times compression ratio

	checkCudaErrors(cudaMalloc((void**)&pCsrFvalue, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrDefault2Right, sizeof(bool) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrFeaStartPos, sizeof(uint) * numFea * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrFeaLen, sizeof(uint) * numFea * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachCsrNodeStartPos, sizeof(uint) * maxNumSN));
	checkCudaErrors(cudaMalloc((void**)&pEachNodeSizeInCsr, sizeof(uint) * maxNumSN));

	checkCudaErrors(cudaMalloc((void**)&preFvalueInsId, sizeof(int) * totalNumFeaValue));
}

void BagCsrManager::reserveCsrSpace(){
	checkCudaErrors(cudaFree(pCsrFvalue));
	checkCudaErrors(cudaFree(pCsrDefault2Right));

	//reserve larger memory
	printf("max num of csr is %u\n", reservedMaxNumCsr);
	checkCudaErrors(cudaMalloc((void**) &pCsrFvalue, sizeof(real) * reservedMaxNumCsr));
	checkCudaErrors(cudaMalloc((void**)&pCsrDefault2Right, sizeof(bool) * reservedMaxNumCsr));
}

//reserve memory for a variable
void BagCsrManager::reserveSpace(MemVector &vec, uint newSize, uint numByteEachValue){
	checkCudaErrors(cudaFree(vec.addr));
	vec.size = newSize;
	vec.reservedSize = newSize * 2;
	checkCudaErrors(cudaMalloc((void**)&vec.addr, numByteEachValue * vec.reservedSize));
}

/** operations on cross variable reused memory **/
uint *BagCsrManager::getMutableCsrLen(){
	PROCESS_ERROR(curNumCsr > 0);
	if(csrLen.reservedSize < curNumCsr)
		reserveSpace(csrLen, curNumCsr, sizeof(uint));
	PROCESS_ERROR(csrLen.addr != NULL);
	return (uint*)csrLen.addr;
}
double *BagCsrManager::getMutableCsrGD(){
	PROCESS_ERROR(curNumCsr > 0);
	if(csrGD.reservedSize < curNumCsr)
		reserveSpace(csrGD, curNumCsr, sizeof(double));

	PROCESS_ERROR(csrGD.addr != NULL);
	return (double*)csrGD.addr;
}
real *BagCsrManager::getMutableCsrHess(){
	PROCESS_ERROR(curNumCsr > 0);
	if(csrHess.reservedSize < curNumCsr * 2)
		reserveSpace(csrHess, curNumCsr * 2, sizeof(real));//reserve 2 times more, for sharing space with CsrFvalueSparse.
	PROCESS_ERROR(csrHess.addr != NULL);
	return (real*)csrHess.addr;
}
real *BagCsrManager::getMutableCsrGain(){
	PROCESS_ERROR(curNumCsr > 0);
	if(csrGain.reservedSize < curNumCsr * 2)
		reserveSpace(csrGain, curNumCsr * 2, sizeof(real));
	PROCESS_ERROR(csrGain.addr != NULL);
	return (real*)csrGain.addr;
}
uint *BagCsrManager::getMutableCsrKey(){
	PROCESS_ERROR(curNumCsr > 0);
	if(pCsrKey.reservedSize < curNumCsr)
		reserveSpace(pCsrKey, curNumCsr, sizeof(uint));
	PROCESS_ERROR(pCsrKey.addr != NULL);
	return (uint*)pCsrKey.addr;
}

uint *BagCsrManager::getMutableCsrStartCurRound(){
	return getMutableCsrKey();
}

uint *BagCsrManager::getMutableCsrMarker(){
	return (uint*)getMutableCsrGain();
}

uint *BagCsrManager::getMutableCsrStart(){
	return getMutableCsrLen();
}

real *BagCsrManager::getMutableCsrFvalueSparse(){
	return getMutableCsrHess();
}

uint *BagCsrManager::getMutableNewCsrLen(){
	return (uint*)getMutableCsrGD();
}

const uint *BagCsrManager::getCsrLen(){
	PROCESS_ERROR(csrLen.addr != NULL);
	return (uint*)csrLen.addr;
}
const double *BagCsrManager::getCsrGD(){
	PROCESS_ERROR(csrGD.addr != NULL);
	return (double*)csrGD.addr;
}
const real *BagCsrManager::getCsrHess(){
	PROCESS_ERROR(csrHess.addr != NULL);
	return (real*)csrHess.addr;
}
const real *BagCsrManager::getCsrGain(){
	PROCESS_ERROR(csrGain.addr != NULL);
	return (real*)csrGain.addr;
}
const uint *BagCsrManager::getCsrKey(){
	PROCESS_ERROR(pCsrKey.addr != NULL);
	return (uint*)pCsrKey.addr;
}
const uint *BagCsrManager::getNewCsrLen(){
	return (uint*)getCsrGD();
}

const real *BagCsrManager::getCsrFvalueSparse(){
	return getCsrHess();
}

const uint *BagCsrManager::getCsrStart(){
	return getCsrLen();
}

const uint *BagCsrManager::getCsrMarker(){
	return (uint*)getCsrGain();
}

const uint *BagCsrManager::getCsrStartCurRound(){
	return getCsrKey();//reuse this memory
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
bool *BagCsrManager::getMutableDefault2Right(){
	PROCESS_ERROR(curNumCsr > 0);
	if(reservedMaxNumCsr < curNumCsr){
		reservedMaxNumCsr = curNumCsr * 2;
		reserveCsrSpace();
	}
	PROCESS_ERROR(pCsrDefault2Right != NULL);
	return pCsrDefault2Right;
}
const real *BagCsrManager::getCsrFvalue(){
	PROCESS_ERROR(pCsrFvalue != NULL);
	return pCsrFvalue;
}
const bool *BagCsrManager::getDefault2Right(){
	PROCESS_ERROR(pCsrDefault2Right != NULL);
	return pCsrDefault2Right;
}

