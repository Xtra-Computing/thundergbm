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
	static int m_numFea;
	static int m_maxNumofSN;
	static long long m_total_copy;

	//pinned memory
	static unsigned int *m_pNumFeaValueEachNode_dh;
	static int *m_pPosToBuffId_dh;

	//device memory
	static unsigned int *pPartitionMarker;
	static unsigned int *m_pnKey;

	//keep track of arranged fvalue and instance ids
	static int* m_pArrangedInsId_d;
	static float_point* m_pArrangedFvalue_d;
	static int* m_pArrangedFeaId_d;

	//histogram based partitioning
	static unsigned int *m_pHistogram_d;
	static unsigned int m_numElementEachThd;
	static unsigned int m_totalNumEffectiveThd;
	static unsigned int *m_pEachNodeStartPos_d;

public:
	//compute indices for each feature value in feature lists
	static void ComputeIdxGPU(int numNode, int maxNumSN, int bagId);

	static void AllocMem(int nNumofExamples, int nNumofFeatures, int maxNumofSpittableNode);
	static void FreeMem();
};



#endif /* INDEXCOMPUTER_H_ */
