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

#include "../Tree/RegTree.h"
#include "../../DeviceHost/TreeNode.h"
#include "../KeyValue.h"
#include "SplitPoint.h"
#include "../../DeviceHost/NodeStat.h"
#include "../GDPair.h"
#include "../../DeviceHost/BaseClasses/BaseSplitter.h"

using std::vector;
using std::map;


class HostSplitter: public BaseSplitter
{
public:
	vector<float_point> m_vPredBuffer;
	vector<float_point> m_vTrueValue;

public:
	virtual string SpliterType(){return "host";}
	//a function for computing the gain of a feature
	virtual void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &tempStat, vector<nodeStat> &lchildStat);
	virtual void SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
				  	  	  const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel);
	virtual void ComputeGD(vector<RegTree> &vTree, vector<vector<KeyValue> > &vvInsSparse);
	void ComputeGDSparse(vector<float_point> &v_fPredValue, vector<float_point> &m_vTrueValue_fixedPos);
};



#endif /* SPLITTER_H_ */
