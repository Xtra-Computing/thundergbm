/*
 * CsrSplit.h
 *
 *  Created on: Jul 11, 2017
 *      Author: zeyi
 */

#ifndef CSRSPLIT_H_
#define CSRSPLIT_H_

#include "../../SharedUtility/DataType.h"

void CsrCompression(int numofSNode, uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos, uint *eachCompressedFeaLen,
		uint *eachNodeSizeInCsr, uint *eachCsrNodeStartPos, real *csrFvalue, double *csrGD_h, real *csrHess_h, uint *eachCsrLen);
__global__ void LoadFvalueInsId(int numIns, const int *pOrgFvalueInsId, int *pNewFvalueInsId, const unsigned int *pDstIndexEachFeaValue, int numFeaValue);
__global__ void newCsrLenFvalue(const int *preFvalueInsId, int numFeaValue, const int *pInsId2Nid, int maxNid,
						  const uint *eachCsrStart, real *csrFvalue, uint numCsr, const uint *preRoundEachCsrFeaStartPos, const uint preRoundNumSN, int numFea,
						  real *eachCsrFvalueSparse, uint *csrNewLen, uint *eachCsrFeaLen, uint *eachNodeSizeInCsr, int numSN, uint *eachNodeFvalue);
__global__ void map2One(const uint *eachCsrFeaLen, uint numCsr, uint *csrMarker);
__global__ void loadDenseCsr(const real *eachCsrFvalueSparse, const uint *eachCsrFeaLen, uint numCsr, uint numCsrThisRound,
							 const uint *csrIdx, real *eachCsrFvalueDense, uint *eachCsrFeaLenDense);
__global__ void compCsrGDHess(const int *preFvalueInsId, uint numFvalue, const uint *eachCsrStart, uint numCsr,
							  const real *pInsGrad, const real *pInsHess, int numIns, double *csrGD, real *csrHess);


#endif /* CSRSPLIT_H_ */
