/*
 * DeviceSplitter.h
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITTER_H_
#define DEVICESPLITTER_H_

#include <vector>

#include "../../pureHost/UpdateOps/NodeStat.h"
#include "../../pureHost/UpdateOps/SplitPoint.h"
#include "../../pureHost/BaseClasses/BaseSplitter.h"

using std::vector;


class DeviceSplitter: public BaseSplitter
{
public:
	virtual void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat);
	static int AssignBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable);

	/**
	 * @brief: return buffer id given a splittable node id
	 */
	static int GetBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable);
};



#endif /* DEVICESPLITTER_H_ */
