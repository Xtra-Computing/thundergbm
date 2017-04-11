/*
 * IndexComputer.h
 *
 *  Created on: 21 Jul 2016
 *      Author: Zeyi Wen
 *		@brief: compute the index of the new node for each feature value in feature lists
 */

#ifndef INDEXCOMPUTER_H_
#define INDEXCOMPUTER_H_

#include "../../DeviceHost/NodeStat.h"

class IndexComputer
{
public:
	//host
	static int m_totalFeaValue;
	static long long *m_pFeaStartPos;
	static int m_numFea;
	static int m_maxNumofSN;
	static long long m_total_copy;

	//pinned memory
	static long long *m_pNumFeaValueEachNode_dh;
	static int *m_pPosToBuffId_dh;

	//device memory
	static unsigned int *pnSparseGatherIdx;
	static unsigned int *pnKey;

public:
	//compute indices for each feature value in feature lists
	static void ComputeIdxGPU(int numNode, int maxNumSN, int bagId);

	static void AllocMem(int nNumofExamples, int nNumofFeatures, int maxNumofSpittableNode);
};



#endif /* INDEXCOMPUTER_H_ */
