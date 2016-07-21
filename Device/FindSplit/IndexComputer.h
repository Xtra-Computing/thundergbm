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

	//pinned memory
	static int *m_insIdToNodeId_dh;
	static int *m_pIndices_dh;
	static int *m_pNumFeaValueEachNode_dh;
	static int *m_pFeaValueStartPosEachNode_dh;
	static int *m_pEachFeaStartPosEachNode_dh;
	static int *m_pEachFeaLenEachNode_dh;

public:
	//compute indices for each feature value in feature lists
	static void ComputeIndex(int numNode, const int *pSNIdToBuffId, int maxNumSN, const int *pBuffVec);

private:
	static int MapBuffIdToPos(int buffId, int minBuffId, const int *pBuffIdToPos);
};



#endif /* INDEXCOMPUTER_H_ */
