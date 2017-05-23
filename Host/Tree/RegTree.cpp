/*
 * RegTree.cpp
 *
 *  Created on: 15 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "RegTree.h"
#include "../../DeviceHost/MyAssert.h"

/**
 * @brief: get the leaf index given a sparse instance
 */
int RegTree::GetLeafIdSparseInstance(vector<real> &ins, map<int, int> &fidToDensePos)
{
	int pid = 0; //node id
	TreeNode *curNode = (*this)[pid];
	while (!curNode->isLeaf())
	{
		int fid = curNode->featureId;
		PROCESS_ERROR(fid >= 0);
		int pos = fidToDensePos[fid];

		if(pos < ins.size())//feature value is available in the dense vector
			pid = curNode->GetNext(ins[pos]);
		else//feature value is stored in the dense vector (due to truncating)
			pid = curNode->GetNext(0);
		curNode = (*this)[pid];
	}

	return pid;
}
