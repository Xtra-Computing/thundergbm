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
						   const uint preRoundNumSN, int numFea, const uint *csrId2SegId, const uint *oldCsrLen,
						   const unsigned char *csrId2Pid, real *eachCsrFvalueSparse, uint *newCsrLen, uint *newCsrFeaLen){
	uint csrId = GLOBAL_TID();//one thread per csr
	if(csrId >= numCsr)
		return;
	uint pid = csrId2Pid[csrId];
	if(pid == LARGE_1B_UCHAR)
		return;
	uint numCsrCurPart;
	uint numCsrPrePartsAhead;//all previous node-features
	uint posInPart;
	uint feaId;
	computeCsrInfo(csrId, preRoundSegStartPos, preRoundNumSN, numFea, numCsr, csrId2SegId,
				  numCsrPrePartsAhead, posInPart, numCsrCurPart, feaId);

	uint basePos = numCsrPrePartsAhead * 2 + posInPart;
	real temp = csrFvalue[csrId];
	eachCsrFvalueSparse[basePos] = temp;
	eachCsrFvalueSparse[basePos + numCsrCurPart] = temp;

	//compute csr length and csr feature length
	uint csrLen = oldCsrLen[csrId] - newCsrLen[basePos];
	CONCHECKER(oldCsrLen[csrId] >= newCsrLen[basePos]);
	uint firstPartId = (pid % 2 == 0) ? pid : (pid - 1);
	if(newCsrLen[basePos] > 0){
		atomicAdd(newCsrFeaLen + firstPartId * numFea + feaId, 1);
	}
	if(csrLen > 0){
		atomicAdd(newCsrFeaLen + (firstPartId + 1) * numFea + feaId, 1);
		newCsrLen[basePos + numCsrCurPart] = csrLen;
	}
//	if(0.369250 == temp){
//		printf("csr len of 0.369250 is %d and %d; old len=%d, csrId=%d\n", newCsrLen[basePos], newCsrLen[basePos + numCsrCurPart], oldCsrLen[csrId], csrId);
//	}
}

__global__ void newCsrLenFvalue(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, const real *csrFvalue, uint numCsr,
						  const uint *preRoundSegStartPos, const uint preRoundNumSN, int numFea, const uint *csrId2SegId,
						  uint *csrNewLen, unsigned char *csrId2Pid){
	//one thread for one fvalue
	uint gTid = GLOBAL_TID();
	extern __shared__ uint csrCounter[];
	__shared__ uint firstCsrId;
	uint tid = threadIdx.x;
	csrCounter[tid] = 0;
	__syncthreads();

	uint csrId;
	if(gTid < numFeaValue)
		RangeBinarySearch(gTid, eachCsrStart, numCsr, csrId);
	CONCHECKER(csrId < numCsr);
	__syncthreads();
	//first csrId
//	if(gTid >= numFeaValue && tid == 0){
//		printf("oh shitt####################################################\n");
//	}
	if(tid == 0)
		firstCsrId = csrId;
	__syncthreads();
	if(gTid < numFeaValue){

		CONCHECKER(csrId >= firstCsrId);

		int insId = preFvalueInsId[gTid];//insId is not -1, as preFvalueInsId is dense.
		int pid = pInsId2Nid[insId] - maxNid - 1;//mapping to new node

//		if(csrId == 2809992 && csrFvalue[csrId] == 0.369250){
//			printf("gpu pid=%d, insId=%d, csrfvalue=%f, left_fv=%f, right_fv=%f, gTid=%d, blkid=%d\n",
//					pid, insId, csrFvalue[csrId], csrFvalue[csrId - 1], csrFvalue[csrId + 1], gTid, blockIdx.y * gridDim.x + blockIdx.x);
//		}

		csrId2Pid[csrId] = pid < 0 ? LARGE_1B_UCHAR : pid;
		if(pid >= 0 && pid % 2 == 0){//not leaf node and it's first part.
			uint counterOffset = csrId - firstCsrId;
//			if(305030 == csrId || 478798 == csrId){
//				printf("first csr id=%d, csrId=%d\n", firstCsrId, csrId);
//			}
			__threadfence();
			uint orgValue = atomicAdd(csrCounter + counterOffset, 1);
//			if(csrFvalue[csrId] == 0.369250){
//				printf("gpu pid=%d, insId=%d, csrfvalue=%f, firstCsrId=%d, cntOffset=%d, orgValue=%d, cnt=%d\n",
//						pid, insId, csrFvalue[csrId], firstCsrId, counterOffset, orgValue, csrCounter[counterOffset]);
//			}
		}
	}

	__syncthreads();
//	if(blockIdx.y * gridDim.x + blockIdx.x == 2216293 && tid == 95){
//		printf("###################################################################### cnt=%d\n", csrCounter[tid]);
//	}
	//compute len of each csr
	if(csrCounter[tid] > 0){
		uint numCsrCurPart;
		uint numCsrPrePartsAhead;
		uint posInPart;
		uint feaId;
		computeCsrInfo(firstCsrId + tid, preRoundSegStartPos, preRoundNumSN, numFea, numCsr, csrId2SegId,
					  numCsrPrePartsAhead, posInPart, numCsrCurPart, feaId);

		atomicAdd(csrNewLen + numCsrPrePartsAhead * 2 + posInPart, csrCounter[tid]);
//		if(blockIdx.y * gridDim.x + blockIdx.x == 2216293 && tid == 95){
//			printf("csrNewLen=%d, numcsrPrePartAhead=%d, posInPart=%d, cnt=%d\n", csrNewLen[numCsrPrePartsAhead * 2 + posInPart], numCsrPrePartsAhead, posInPart, csrCounter[tid]);
//		}
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
	if(eachCsrFeaLen[gTid] > 0){
		uint idx = csrIdx[gTid] - 1;//inclusive scan is used to compute indices.
		CONCHECKER(csrIdx[gTid] <= numCsrThisRound);
		eachCsrFeaLenDense[idx] = eachCsrFeaLen[gTid];
		eachCsrFvalueDense[idx] = eachCsrFvalueSparse[gTid];
	}
}

__global__ void ComputeGDHess(const uint *pCsrLen, const uint *pCsrStartPos, const real *pInsGD, const real *pInsHess,
						  const int *pInsId, double *csrGD, real *csrHess){
	uint csrId = blockIdx.x;
	uint tid = threadIdx.x;
	extern __shared__ double pGD[];
	double *pHess = (pGD + blockDim.x);
	pHess[tid] = 0;
	__syncthreads();

	uint csrLen = pCsrLen[csrId];
	uint csrStart = pCsrStartPos[csrId];

	//load to shared memory
	int i = tid;
	double tempHess = 0;
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
		CONCHECKER(pHess[0] > 0);
		csrHess[csrId] = pHess[0];
		csrGD[csrId] = pGD[0];
	}
}
