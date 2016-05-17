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
#include "../pureHost/GDPair.h"
#include "../pureHost/UpdateOps/NodeStat.h"
#include "../pureHost/UpdateOps/SplitPoint.h"

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

public:

	//has an identical version in device @splitAll
	static int AssignHashValue(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable, bool &bIsNew);

	/**
	 * @brief: return buffer id given a splittable node id
	 */
	static int GetBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable);
};



#endif /* PREPARATOR_H_ */
