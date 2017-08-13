/*
 * BagCsrManager.h
 *
 *  Created on: Jul 23, 2017
 *      Author: zeyi
 */

#ifndef BAGCSRMANAGER_H_
#define BAGCSRMANAGER_H_

#include "../../SharedUtility/DataType.h"
#include "../../SharedUtility/memVector.h"

class BagCsrManager{
public:
	static uint *pEachCsrFeaStartPos;
	static uint *pEachCsrFeaLen;
	static uint *pEachCsrNodeStartPos;
	static uint *pEachNodeSizeInCsr;
	static int *preFvalueInsId;
	static uint curNumCsr;
private:
	static uint reservedMaxNumCsr;
	static MemVector csrLen;
	static MemVector csrKey;
	static real *pCsrFvalue;
	void reserveCsrSpace();

public:
	BagCsrManager(int numFea, int maxNumSN, uint totalNumFeaValue);
	uint *getMutableCsrLen();
	real *getMutableCsrHess();
	real *getMutableCsrFvalue();
	uint *getMutableCsrKey();

	uint *getMutableCsrStart();
	const uint *getCsrStart();
	const uint *getCsrLen();
	const real *getCsrHess();
	const real *getCsrFvalue();
	const uint *getCsrKey();
};



#endif /* BAGCSRMANAGER_H_ */
