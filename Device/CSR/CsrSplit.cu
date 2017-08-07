/*
 * CsrSplit.cu
 *
 *  Created on: Jul 11, 2017
 *      Author: zeyi
 */

#include "CsrSplit.h"
#include "../../SharedUtility/CudaMacro.h"
#include "../../SharedUtility/binarySearch.h"

/**
 * @brief: efficient best feature finder
 */
__global__ void LoadFvalueInsId(int numIns, const int *pOrgFvalueInsId, int *pNewFvalueInsId, const unsigned int *pDstIndexEachFeaValue, int numFeaValue)
{
	//one thread loads one value
	int gTid = GLOBAL_TID();

	if(gTid >= numFeaValue)//thread has nothing to load; note that "numFeaValue" needs to be the length of whole dataset
		return;

	//index for scatter
	uint idx = pDstIndexEachFeaValue[gTid];
	if(idx == LARGE_4B_UINT)//instance is in a leaf node
		return;

	CONCHECKER(idx < numFeaValue);

	//scatter: store the feature value ins id.
	CONCHECKER(numIns >= pOrgFvalueInsId[gTid] && pOrgFvalueInsId[gTid] >= 0);
	pNewFvalueInsId[idx] = pOrgFvalueInsId[gTid];
}

__device__ void computeCsrInfo(uint csrId, const uint *preRoundSegStartPos, const uint preRoundNumSN, int numFea, uint numCsr, const uint *csrId2SegId,
							  uint &numCsrPrePartsAhead, uint &posInPart, uint &numCsrCurPart, uint &feaId){
	uint segId = csrId2SegId[csrId];
	uint prePid = segId / numFea;
	uint prePartStartPos = preRoundSegStartPos[prePid * numFea];
	feaId = segId % numFea;
	numCsrPrePartsAhead = prePartStartPos;
	posInPart = csrId - numCsrPrePartsAhead;//id in the partition
	ECHECKER(csrId >= numCsrPrePartsAhead);
	if(prePid == preRoundNumSN - 1)
		numCsrCurPart = numCsr - prePartStartPos;
	else
		numCsrCurPart = preRoundSegStartPos[(prePid + 1) * numFea] - prePartStartPos;
}

__global__ void fillFvalue(const real *csrFvalue, uint numCsr, const uint *preRoundSegStartPos,
						   const uint preRoundNumSN, int numFea, const uint *csrId2SegId,
						   real *eachCsrFvalueSparse){
	uint csrId = GLOBAL_TID();//one thread per csr
	if(csrId >= numCsr)
		return;
	uint numCsrCurPart;
	uint numCsrPrePartsAhead;
	uint posInPart;
	uint feaId;
	computeCsrInfo(csrId, preRoundSegStartPos, preRoundNumSN, numFea, numCsr, csrId2SegId,
				  numCsrPrePartsAhead, posInPart, numCsrCurPart, feaId);

	uint basePos = numCsrPrePartsAhead * 2 + posInPart;
	real temp = csrFvalue[csrId];
	eachCsrFvalueSparse[basePos] = temp;
	eachCsrFvalueSparse[basePos + numCsrCurPart] = temp;
}

__global__ void newCsrLenFvalue(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, const real *csrFvalue, uint numCsr,
						  const uint *preRoundSegStartPos, const uint preRoundNumSN, int numFea, const uint *csrId2SegId,
						  uint *csrNewLen, uint *eachNewSegLen){
	//one thread for one fvalue
	uint gTid = GLOBAL_TID();
	extern __shared__ uint csrCounter[];
	uint *pCsrId2Pid = csrCounter + blockDim.x * 2;
	__shared__ uint firstCsrId;
	uint tid = threadIdx.x;
	csrCounter[tid] = 0;
	csrCounter[tid + blockDim.x] = 0;
	pCsrId2Pid[tid] = LARGE_4B_UINT;
	pCsrId2Pid[tid + blockDim.x] = LARGE_4B_UINT;
	__syncthreads();
	if(gTid < numFeaValue){//thread has nothing to do
		uint csrId;
		RangeBinarySearch(gTid, eachCsrStart, numCsr, csrId);
		CONCHECKER(csrId < numCsr);
		//first csrId
		if(tid == 0)
			firstCsrId = csrId;
		__syncthreads();
		CONCHECKER(csrId >= firstCsrId);

		int insId = preFvalueInsId[gTid];//insId is not -1, as preFvalueInsId is dense.
		int pid = pInsId2Nid[insId] - maxNid - 1;//mapping to new node
		if(pid >= 0){//not leaf node
			uint counterPosInShared = csrId - firstCsrId;
			if(pid % 2 == 0){
				atomicAdd(csrCounter + counterPosInShared, 1);
				pCsrId2Pid[counterPosInShared] = pid;
			}
			else{
				atomicAdd(csrCounter + blockDim.x + counterPosInShared, 1);
				pCsrId2Pid[counterPosInShared + blockDim.x] = pid;
			}
		}
	}
	__syncthreads();
	//compute len of each csr
	if(csrCounter[tid] == 0 && csrCounter[tid + blockDim.x] == 0)
		return;


	uint numCsrCurPart;
	uint numCsrPrePartsAhead;
	uint posInPart;
	uint feaId;
	computeCsrInfo(firstCsrId + tid, preRoundSegStartPos, preRoundNumSN, numFea, numCsr, csrId2SegId,
				  numCsrPrePartsAhead, posInPart, numCsrCurPart, feaId);

	CONCHECKER(feaId < numFea);
	if(csrCounter[tid] > 0){
		uint orgValue = atomicAdd(csrNewLen + numCsrPrePartsAhead * 2 + posInPart, csrCounter[tid]);
		if(orgValue == 0){
			CONCHECKER(pCsrId2Pid[tid] < 256);
			atomicAdd(eachNewSegLen + pCsrId2Pid[tid] * numFea + feaId, 1);
		}
	}
	if(csrCounter[tid + blockDim.x] > 0){
		uint orgValue = atomicAdd(csrNewLen + numCsrPrePartsAhead * 2 + numCsrCurPart + posInPart, csrCounter[tid + blockDim.x]);
		if(orgValue == 0){
			CONCHECKER(pCsrId2Pid[tid + blockDim.x] < 256);
			atomicAdd(eachNewSegLen + pCsrId2Pid[tid + blockDim.x] * numFea + feaId, 1);
		}
	}
}

__global__ void newCsrLenFvalue2(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, const real *csrFvalue, uint numCsr,
						  const uint *preRoundSegStartPos, const uint preRoundNumSN, int numFea, const uint *csrId2SegId,
						  uint *csrNewLen, uint *eachNewSegLen){
	//one thread for one fvalue
	uint gTid = GLOBAL_TID();
	extern __shared__ uint csrCounter[];
	uint *pCsrId2Pid = csrCounter + blockDim.x * 2;
	__shared__ uint firstCsrId;
	uint tid = threadIdx.x;
	csrCounter[tid] = 0;
	csrCounter[tid + blockDim.x] = 0;
	pCsrId2Pid[tid] = LARGE_4B_UINT;
	pCsrId2Pid[tid + blockDim.x] = LARGE_4B_UINT;
	__syncthreads();
	if(gTid < numFeaValue){//thread has value to load
		uint csrId;
		RangeBinarySearch(gTid, eachCsrStart, numCsr, csrId);
		CONCHECKER(csrId < numCsr);
		//first csrId
		if(tid == 0)
			firstCsrId = csrId;
		__syncthreads();
		CONCHECKER(csrId >= firstCsrId);

		int insId = preFvalueInsId[gTid];//insId is not -1, as preFvalueInsId is dense.
		int pid = pInsId2Nid[insId] - maxNid - 1;//mapping to new node
		if(pid >= 0){//not leaf node
			uint counterOffset = blockDim.x * (pid % 2) + csrId - firstCsrId;
			atomicAdd(csrCounter + counterOffset, 1);
			pCsrId2Pid[counterOffset] = pid;
		}
	} else {
		__syncthreads();
	}
	__syncthreads();
	//compute len of each csr
	uint localCsrId = tid / 2;
	int offsetFlag = tid % 2;
	uint offset = blockDim.x * offsetFlag;
	while(localCsrId < blockDim.x){
		uint counterPos = localCsrId + offset;
		if(csrCounter[counterPos] > 0){
			uint numCsrCurPart;
			uint numCsrPrePartsAhead;
			uint posInPart;
			uint feaId;
			computeCsrInfo(firstCsrId + localCsrId, preRoundSegStartPos, preRoundNumSN, numFea, numCsr, csrId2SegId,
						  numCsrPrePartsAhead, posInPart, numCsrCurPart, feaId);

			uint orgValue = atomicAdd(csrNewLen + numCsrPrePartsAhead * 2 + numCsrCurPart * offsetFlag + posInPart, csrCounter[counterPos]);
			if(orgValue == 0){
				CONCHECKER(pCsrId2Pid[counterPos] < 256);
				atomicAdd(eachNewSegLen + pCsrId2Pid[counterPos] * numFea + feaId, 1);
			}
		}
		localCsrId += (blockDim.x/2);
	}
}

__global__ void map2One(const uint *eachCsrLen, uint numCsr, uint *csrMarker){
	uint gTid = GLOBAL_TID();
	if(gTid >= numCsr)
		return;
	if(eachCsrLen[gTid] > 0)
		csrMarker[gTid] = 1;
	else
		csrMarker[gTid] = 0;
}

__global__ void loadDenseCsr(const real *eachCsrFvalueSparse, const uint *eachCsrFeaLen, uint numCsr, uint numCsrThisRound,
							 const uint *csrIdx, real *eachCsrFvalueDense, uint *eachCsrFeaLenDense){
	uint gTid = GLOBAL_TID();
	if(gTid >= numCsr)
		return;
	if(eachCsrFeaLen[gTid] != 0){
		uint idx = csrIdx[gTid] - 1;//inclusive scan is used to compute indices.
		CONCHECKER(csrIdx[gTid] <= numCsrThisRound);
		eachCsrFeaLenDense[idx] = eachCsrFeaLen[gTid];
		eachCsrFvalueDense[idx] = eachCsrFvalueSparse[gTid];
	}
}

__global__ void compCsrGDHess(const int *preFvalueInsId, uint numUsefulFvalue, const uint *eachCsrStart, uint numCsr,
							  const real *pInsGrad, const real *pInsHess, int numIns,
							  double *csrGD, real *csrHess){
	uint gTid = GLOBAL_TID();
	if(gTid >= numUsefulFvalue)
		return;
	uint csrId = numCsr;
	RangeBinarySearch(gTid, eachCsrStart, numCsr, csrId);
	CONCHECKER(csrId < numCsr);
	int insId = preFvalueInsId[gTid];
	CONCHECKER(insId >= 0 && insId < numIns);
	double temp = pInsGrad[insId];
	atomicAdd(csrGD + csrId, temp);
	atomicAdd(csrHess + csrId, pInsHess[insId]);
}

__global__ void ComputeGD(const uint *pCsrLen, const uint *pCsrStartPos, const real *pInsGD, const real *pInsHess,
						  const int *pInsId, double *csrGD, real *csrHess){
	uint csrId = blockIdx.x;
	uint tid = threadIdx.x;
	extern __shared__ double pGD[];
	real *pHess = (real*)(pGD + blockDim.x);
	uint csrLen = pCsrLen[csrId];
	uint csrStart = pCsrStartPos[csrId];

	//load to shared memory
	int i = tid;
	real tempHess = 0;
	double tempGD = 0;
	while(i < csrLen){
		int insId = pInsId[csrStart + i];
		tempHess += pInsHess[insId];
		tempGD += pInsGD[insId];
		i += blockDim.x;
	}
	pHess[tid] = tempHess;
	pGD[tid] = tempGD;

	//reduction
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if(tid < offset) {
			pHess[tid] += pHess[tid + offset];
			pGD[tid] += pGD[tid + offset];
		}
		__syncthreads();
	}
	if(tid == 0){
		csrHess[csrId] = pHess[0];
		csrGD[csrId] = pGD[0];
	}
}
