/*
 * BagCsrManager.h
 *
 *  Created on: Jul 23, 2017
 *      Author: zeyi
 */

#ifndef BAGCSRMANAGER_H_
#define BAGCSRMANAGER_H_

#include "../../SharedUtility/DataType.h"

class BagCsrManager{
public:
	static uint *pEachCsrFeaStartPos;
	static uint *pEachCsrFeaLen;
	static uint *pEachCsrNodeStartPos;
	static uint *pEachNodeSizeInCsr;
private:
	static uint curNumCsr;
	static uint reservedMaxNumCsr;
	static uint *pCsrLen;
	static double *pCsrGD;
	static real *pCsrHess;
	static real *pCsrFvalue;
	void reserveSpace();

public:
	BagCsrManager();
	uint *getMutableCsrLen(uint numCsr);
	double *getMutableCsrGD(uint numCsr);
	real *getMutableCsrHess(uint numCsr);
	real *getMutableCsrFvalue(uint numCsr);

	const uint *getCsrLen();
	const double *getCsrGD();
	const real *getCsrHess();
	const real *getCsrFvalue();
};



#endif /* BAGCSRMANAGER_H_ */
