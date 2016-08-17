/*
 * FindFeaMemManager.h
 *
 *  Created on: 16 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef FINDFEAMEMMANAGER_H_
#define FINDFEAMEMMANAGER_H_

#include "../../DeviceHost/DefineConst.h"

class FFMemManager
{
public:
	//memory for the tree
	static int maxNumofSNodeInFF;

	//for dense array
	static float_point *pGDEachFeaValue, *pHessEachFeaValue, *pDenseFeaValue;
	static float_point *pGDPrefixSum, *pHessPrefixSum;
	static float_point *pGainEachFeaValue;
	static int m_totalNumFeaValue;
	static float_point *pfLocalBestGain_d;
	static int *pnLocalBestGainKey_d;
	static float_point *pfGlobalBestGain_d;
	static int *pnGlobalBestGainKey_d;
	//corresponding to pinned memory
	static int *m_pIndices_d;
	static long long *m_pFeaValueStartPosEachNode_d;
	static long long *m_pNumFeaValueEachNode_d;
	static long long *m_pEachFeaStartPosEachNode_d;
	static int *m_pEachFeaLenEachNode_d;

private:
	static long long m_totalEleInWholeBatch;

public:
	int getMaxNumofSN(int numofValuesInABatch, int maxNumofNode);
	void allocMemForFindFea(int numofValuesInABatch, int maxNumofValuePerFea, int maxNumofFea, int maxNumofSN);

	void resetMemForFindFea();
	void freeMemForFindFea();
};



#endif /* FINDFEAMEMMANAGER_H_ */
