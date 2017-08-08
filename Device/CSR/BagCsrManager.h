/*
 * BagCsrManager.h
 *
 *  Created on: Jul 23, 2017
 *      Author: zeyi
 */

#ifndef BAGCSRMANAGER_H_
#define BAGCSRMANAGER_H_

#include "../../SharedUtility/DataType.h"

struct MemVector{
	void *addr = NULL;
	uint size = 0;
	uint reservedSize = 0;
};

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
	static MemVector csrGD;
	static MemVector csrHess;
	static MemVector csrGain;
	static MemVector csrKey;
	static MemVector csrDefault2Right;
	static real *pCsrFvalue;
	void reserveCsrSpace();
	void reserveSpace(MemVector &vec, uint newSize, uint numByteEachValue);

public:
	BagCsrManager(int numFea, int maxNumSN, uint totalNumFeaValue);
	uint *getMutableCsrLen();
	double *getMutableCsrGD();
	real *getMutableCsrHess();
	real *getMutableCsrFvalue();
	real *getMutableCsrGain();
	uint *getMutableCsrKey();
	bool *getMutableDefault2Right();

	uint *getMutableNewCsrLen();
	real *getMutableCsrFvalueSparse();
	uint *getMutableCsrStart();
	uint *getMutableCsrMarker();
	uint *getMutableCsrStartCurRound();
	unsigned char *getMutableCsrId2Pid();
	uint *getMutableCsrOldLen();
	const uint *getNewCsrLen();
	const real *getCsrFvalueSparse();
	const uint *getCsrStart();
	const uint *getCsrMarker();
	const uint *getCsrStartCurRound();
	const unsigned char *getCsrId2Pid();
	const uint *getCsrOldLen();
	const uint *getCsrLen();
	const double *getCsrGD();
	const real *getCsrHess();
	const real *getCsrFvalue();
	const real *getCsrGain();
	const uint *getCsrKey();
	const bool *getDefault2Right();
};



#endif /* BAGCSRMANAGER_H_ */
