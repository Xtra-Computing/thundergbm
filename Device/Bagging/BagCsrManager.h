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
	static int *preFvalueInsId;
	static uint curNumCsr;
private:
	static uint reservedMaxNumCsr;
	static uint *pCsrLen;
	static double *pCsrGD;
	static real *pCsrHess;
	static real *pCsrFvalue;
	static bool *pCsrDefault2Right;
	static real *pCsrGain;
	static uint *pCsrKey;
	void reserveSpace();

public:
	BagCsrManager(int numFea, int maxNumSN, uint totalNumFeaValue);
	uint *getMutableCsrLen();
	double *getMutableCsrGD();
	real *getMutableCsrHess();
	real *getMutableCsrFvalue();
	real *getMutableCsrGain();
	uint *getMutableCsrKey();
	bool *getMutableDefault2Right();

	const uint *getCsrLen();
	const double *getCsrGD();
	const real *getCsrHess();
	const real *getCsrFvalue();
	const real *getCsrGain();
	const uint *getCsrKey();
	const bool *getDefault2Right();
};



#endif /* BAGCSRMANAGER_H_ */
