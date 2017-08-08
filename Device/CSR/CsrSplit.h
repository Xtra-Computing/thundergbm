/*
 * CsrSplit.h
 *
 *  Created on: Jul 11, 2017
 *      Author: zeyi
 */

#ifndef CSRSPLIT_H_
#define CSRSPLIT_H_

#include "../../SharedUtility/DataType.h"

__global__ void LoadFvalueInsId(int numIns, const int *pOrgFvalueInsId, int *pNewFvalueInsId, const unsigned int *pDstIndexEachFeaValue, int numFeaValue);
__global__ void fillFvalue(const real *csrFvalue, uint numCsr, const uint *preRoundSegStartPos,
						   const uint preRoundNumSN, int numFea, const uint *csrId2SegId, const uint *oldCsrLen,
						   const unsigned char *csrId2Pid, real *eachCsrFvalueSparse, uint *newCsrLen, uint *newCsrFeaLen);
__global__ void newCsrLenFvalue2(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, const real *csrFvalue, uint numCsr, const uint *preRoundEachCsrFeaStartPos, const uint preRoundNumSN, int numFea,
						  const uint *csrId2SegId, uint *csrNewLen, uint *eachCsrFeaLen);
__global__ void newCsrLenFvalue3(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, const real *csrFvalue, uint numCsr, const uint *preRoundEachCsrFeaStartPos, const uint preRoundNumSN, int numFea,
						  const uint *csrId2SegId, uint *csrNewLen, unsigned char *csrId2Pid);
__global__ void map2One(const uint *eachCsrFeaLen, uint numCsr, uint *csrMarker);
__global__ void loadDenseCsr(const real *eachCsrFvalueSparse, const uint *eachCsrFeaLen, uint numCsr, uint numCsrThisRound,
							 const uint *csrIdx, real *eachCsrFvalueDense, uint *eachCsrFeaLenDense);
__global__ void compCsrGDHess(const int *preFvalueInsId, uint numFvalue, const uint *eachCsrStart, uint numCsr,
							  const real *pInsGrad, const real *pInsHess, int numIns, double *csrGD, real *csrHess);
__global__ void ComputeGD(const uint *pCsrLen, const uint *pCsrStartPos, const real *pInsGD, const real *pInsHess,
						  const int *pInsId, double *csrGD, real *csrHess);

#endif /* CSRSPLIT_H_ */
