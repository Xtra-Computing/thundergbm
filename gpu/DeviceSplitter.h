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
	void FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat);

	static __device__ double CalGain(const nodeStat &parent, const nodeStat &r_child, float_point &l_child_GD,
									 float_point &l_child_Hess, float_point &lambda);
	/**
	 * @brief: return buffer id given a splittable node id
	 */
	static __device__ int GetBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable);
	static __device__ bool UpdateSplitPoint(SplitPoint &curBest, double fGain, double fSplitValue, int nFeatureId);

	static __device__ void UpdateLRStat(nodeStat &RChildStat, nodeStat &LChildStat, nodeStat &TempRChildStat,
										float_point &grad, float_point &hess);
	static __device__ bool NeedUpdate(float_point &RChildHess, float_point &LChildHess);
	static __device__ void UpdateSplitInfo(nodeStat &snStat, SplitPoint &bestSP, nodeStat &RChildStat, nodeStat &LChildStat,
										 nodeStat &TempRChildStat, float_point &tempGD, float_point &temHess,
										 float_point &lambda, float_point &sv, int &featureId);

};



#endif /* DEVICESPLITTER_H_ */
