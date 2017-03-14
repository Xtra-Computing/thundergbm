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
	static int *m_pInsId;
	static int m_totalFeaValue;
	static long long *m_pFeaStartPos;
	static int m_numFea;
	static int m_maxNumofSN;
	static int *m_pIndexCounterEachNode;
	static long long m_total_copy;

	//pinned memory
	static int *m_insIdToNodeId_dh;
	static int *m_pIndices_dh;
	static long long *m_pFeaValueStartPosEachNode_dh;
	static long long *m_pNumFeaValueEachNode_dh;
	static long long *m_pEachFeaStartPosEachNode_dh;
	static int *m_pEachFeaLenEachNode_dh;
	static int *m_pBuffIdToPos_dh;
	static int *m_pPosToBuffId_dh;


	static unsigned int *m_pEachFeaStartPos_dh;
	static unsigned int *m_pnGatherIdx;


public:
	//compute indices for each feature value in feature lists
	static void ComputeIndex(int numNode, const int *pSNIdToBuffId, int maxNumSN, const int *pBuffVec);
	static void ComputeIdxGPU(int numNode, int maxNumSN, const int *pBuffVec);

	static void AllocMem(int nNumofExamples, int nNumofFeatures, int maxNumofSpittableNode);
};



#endif /* INDEXCOMPUTER_H_ */
