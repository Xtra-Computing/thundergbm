/*
 * BagOrgManager.h
 *
 *  Created on: Jul 24, 2017
 *      Author: zeyi
 */

#ifndef BAGORGMANAGER_H_
#define BAGORGMANAGER_H_

#include "../../SharedUtility/DataType.h"

class BagOrgManager{
public:
	static real *m_pdDFeaValue;
	static real *m_pDenseFValueEachBag;
	static double *m_pdGDPrefixSumEachBag;
	static real *m_pHessPrefixSumEachBag;
	static real *m_pGainEachFvalueEachBag;
	static uint *m_pnKey_d;
	static uint *m_pnTid2Fid;
	static bool needCopy;
	BagOrgManager(uint numFeaValue, int numBag);
};



#endif /* BAGORGMANAGER_H_ */
