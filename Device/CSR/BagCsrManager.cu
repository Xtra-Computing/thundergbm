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
MemVector BagCsrManager::csrHess;
MemVector BagCsrManager::csrGain; //shared with csrMarker and old
MemVector BagCsrManager::csrKey; //shared with pCsrStartCurRound
real *BagCsrManager::pCsrFvalue = NULL;
MemVector BagCsrManager::csrDefault2Right; //shared with csrId2Pid

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

//reserve memory for a variable
void BagCsrManager::reserveSpace(MemVector &vec, uint newSize, uint numByteEachValue){
	if(vec.addr != NULL){
		checkCudaErrors(cudaFree(vec.addr));
	}
	vec.size = newSize;
	vec.reservedSize = newSize * 1.5;
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
real *BagCsrManager::getMutableCsrHess(){
	PROCESS_ERROR(curNumCsr > 0);
	if(csrHess.reservedSize < curNumCsr)
		reserveSpace(csrHess, curNumCsr, sizeof(real));
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
	if(csrKey.reservedSize < curNumCsr)
		reserveSpace(csrKey, curNumCsr, sizeof(uint));
	PROCESS_ERROR(csrKey.addr != NULL);
	return (uint*)csrKey.addr;
}
bool *BagCsrManager::getMutableDefault2Right(){
	PROCESS_ERROR(curNumCsr > 0);
	if(csrDefault2Right.reservedSize < curNumCsr)
		reserveSpace(csrDefault2Right, curNumCsr, sizeof(bool));
	PROCESS_ERROR(csrDefault2Right.addr != NULL);
	return (bool*)csrDefault2Right.addr;
}

uint *BagCsrManager::getMutableCsrStartCurRound(){
	return getMutableCsrKey();
}
unsigned char *BagCsrManager::getMutableCsrId2Pid(){
	return (unsigned char*)getMutableDefault2Right();
}
uint *BagCsrManager::getMutableCsrMarker(){
	return (uint*)getMutableCsrGain();
}

uint *BagCsrManager::getMutableCsrStart(){
	return getMutableCsrLen();
}

uint *BagCsrManager::getMutableCsrOldLen(){
	return (uint*)getMutableCsrGain();
}

const uint *BagCsrManager::getCsrLen(){
	PROCESS_ERROR(csrLen.addr != NULL);
	return (uint*)csrLen.addr;
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
	PROCESS_ERROR(csrKey.addr != NULL);
	return (uint*)csrKey.addr;
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
const unsigned char *BagCsrManager::getCsrId2Pid(){
	return (unsigned char*)getDefault2Right();
}
const uint *BagCsrManager::getCsrOldLen(){
	return (uint*)getCsrGain();
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
const bool *BagCsrManager::getDefault2Right(){
	PROCESS_ERROR(csrDefault2Right.addr != NULL);
	return (bool*)csrDefault2Right.addr;
}

