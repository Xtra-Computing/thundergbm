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

#include "../pureHost/Tree/RegTree.h"
#include "../pureHost/Tree/TreeNode.h"
#include "../pureHost/KeyValue.h"
#include "../pureHost/UpdateOps/NodeStat.h"
#include "../pureHost/UpdateOps/SplitPoint.h"
#include "../pureHost/UpdateOps/BaseSplitter.h"

using std::vector;

typedef double float_point;

class DeviceSplitter: public BaseSplitter
{
public:
	virtual void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat);
};



#endif /* DEVICESPLITTER_H_ */
