/*
 * CsrCompressor.h
 *
 *  Created on: Jul 25, 2017
 *      Author: zeyi
 */

#ifndef CSRCOMPRESSOR_H_
#define CSRCOMPRESSOR_H_

#include "../../SharedUtility/DataType.h"

class CsrCompressor
{
private:
	static uint *eachFeaLenEachNode_h;
	static uint *eachFeaStartPosEachNode_h;
	static uint *eachCsrFeaStartPos_h;
	static uint *eachCompressedFeaLen_h;
	static uint *eachCsrLen_h;
	static uint eachNodeSizeInCsr_h;
	static real *csrFvalue_h;
	static int *insId_h;
	static uint totalOrgNumCsr;

	static uint *pCsrFeaStartPos_d;
	static uint *pCsrFeaLen_d;
	static uint *pCsrLen_d;
	static uint eachNodeSizeInCsr_d;
	static real *pCsrFvalue_d;
	static uint *pCsrStart_d;
public:
	static bool bUseCsr;
	static real *pOrgFvalue;
public:
	CsrCompressor();
	void CsrCompression(uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos, uint *eachCompressedFeaLen,
		uint *eachNodeSizeInCsr, uint *eachCsrNodeStartPos);
	const real *getOrgCsrFvalue(){return pCsrFvalue_d;}
	const uint *getOrgCsrStart(){return pCsrStart_d;}
	uint getNumOrgCsr(){return totalOrgNumCsr;}
};



#endif /* CSRCOMPRESSOR_H_ */
