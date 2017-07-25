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

__global__ void newCsrLenFvalue(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, const real *csrFvalue, uint numCsr, const uint *preRoundSegStartPos, const uint preRoundNumSN, int numFea,
						  real *eachCsrFvalueSparse, uint *csrNewLen, uint *eachNewSegLen, uint *eachNodeSizeInCsr, int numSN){
	//one thread for one fvalue
	uint gTid = GLOBAL_TID();
	if(gTid >= numFeaValue)//thread has nothing to do
		return;

	int insId = preFvalueInsId[gTid];//insId is not -1, as preFvalueInsId is dense.
	if(pInsId2Nid[insId] <= maxNid)//leaf node
		return;
	int pid = pInsId2Nid[insId] - maxNid - 1;//mapping to new node
	CONCHECKER(pid < numSN && pid >= 0);
	uint csrId = numCsr;
	RangeBinarySearch(gTid, eachCsrStart, numCsr, csrId);
	CONCHECKER(csrId < numCsr);
	uint segId = numFea * preRoundNumSN;
	RangeBinarySearch(csrId, preRoundSegStartPos, numFea * preRoundNumSN, segId);
	uint prePid = segId / numFea;
	uint prePartStartPos = preRoundSegStartPos[prePid * numFea];
	uint numCsrPrePartsAhead = prePartStartPos;
	uint numCsrCurPart;
	if(prePid == preRoundNumSN - 1)
		numCsrCurPart = numCsr - prePartStartPos;
	else
		numCsrCurPart = preRoundSegStartPos[(prePid + 1) * numFea] - prePartStartPos;
	uint posInPart = csrId - numCsrPrePartsAhead;//id in the partition
	uint orgValue;
	//compute len of each csr
	if(pid % 2 == 1){
		orgValue = atomicAdd(csrNewLen + numCsrPrePartsAhead * 2 + numCsrCurPart + posInPart, 1);
		if(orgValue == 0)
			eachCsrFvalueSparse[numCsrPrePartsAhead * 2 + numCsrCurPart + posInPart] = csrFvalue[csrId];
	}
	else{
		orgValue = atomicAdd(csrNewLen + numCsrPrePartsAhead * 2 + posInPart, 1);
		if(orgValue == 0)
			eachCsrFvalueSparse[numCsrPrePartsAhead * 2 + posInPart] = csrFvalue[csrId];
	}

	//compute len of each segment
	if(orgValue == 0){
		uint feaId = segId % numFea;
		CONCHECKER(feaId < numFea);
		uint tempLen = atomicAdd(eachNewSegLen + pid * numFea + feaId, 1);
		atomicAdd(eachNodeSizeInCsr + pid, 1);
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
