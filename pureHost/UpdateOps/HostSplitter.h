/*
 * Splitter.h
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: a class for splitting a node
 */

#ifndef SPLITTER_H_
#define SPLITTER_H_

#include <vector>
#include <map>

#include "../Tree/RegTree.h"
#include "../Tree/TreeNode.h"
#include "../KeyValue.h"
#include "SplitPoint.h"
#include "NodeStat.h"
#include "../GDPair.h"
#include "../BaseClasses/BaseSplitter.h"

using std::vector;
using std::map;


class HostSplitter: public BaseSplitter
{
public:
	//a function for computing the gain of a feature
	virtual void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &tempStat, vector<nodeStat> &lchildStat);

};



#endif /* SPLITTER_H_ */
