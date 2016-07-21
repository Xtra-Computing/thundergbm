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

#include "../../Host/UpdateOps/SplitPoint.h"
#include "../../DeviceHost/BaseClasses/BaseSplitter.h"
#include "../../DeviceHost/NodeStat.h"

using std::vector;


class DeviceSplitter: public BaseSplitter
{
public:
	double total_scan_t;
	double total_com_gain_t;
	double total_fix_gain_t;
	double total_fill_gd_t;
	double total_search_t;
public:
	virtual string SpliterType(){return "device";}

	virtual void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat);
	virtual void SplitAll(vector<TreeNode*> &splittableNode, const vector<SplitPoint> &vBest, RegTree &tree, int &m_nNumofNode,
				  	  	  const vector<nodeStat> &rchildStat, const vector<nodeStat> &lchildStat, bool bLastLevel);
	virtual void ComputeGD(vector<RegTree> &vTree, vector<vector<KeyValue> > & vvInsSparse);

};



#endif /* DEVICESPLITTER_H_ */
