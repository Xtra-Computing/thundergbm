/*
 * Preparator.h
 *
 *  Created on: 11 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef PREPARATOR_H_
#define PREPARATOR_H_

#include <vector>
#include <map>
#include "../Host/GDPair.h"
#include "../Host/UpdateOps/SplitPoint.h"
#include "../DeviceHost/NodeStat.h"

using std::vector;
using std::map;

class DataPreparator
{
public:
	static int *m_pSNIdToBuffIdHost;//use in two functions
	static int *m_pUsedFIDMap;//(memset require!) for mapping used feature ids
public:
	static void PrepareGDHess(const vector<gdpair> &m_vGDPair_fixedPos);
	static void PrepareSNodeInfo(const map<int, int> &mapNodeIdToBufferPos, const vector<nodeStat> &m_nodeStat);
	static void CopyBestSplitPoint(const map<int, int> &mapNodeIdToBufferPos, vector<SplitPoint> &vBest,
								   vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat);

	static void ReleaseMem()
	{
		if(m_pSNIdToBuffIdHost != NULL)
			delete []m_pSNIdToBuffIdHost;
	}

	//convert a vector to an array
	template<class T>
	static void VecToArray(vector<T> vec, T* arr)
	{
		int numofEle = vec.size();
		for(int i = 0; i < numofEle; i++)
		{
			arr[i] = vec[i];
		}
	}
};



#endif /* PREPARATOR_H_ */
