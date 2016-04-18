/*
 * RegTree.cpp
 *
 *  Created on: 15 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <assert.h>
#include "RegTree.h"

/**
 * \brief get the leaf index
 * \param feats dense feature vector, if the feature is missing the field is set to NaN
 * \param root_gid starting root index of the instance
 * \return the leaf index of the given feature
*/
int RegTree::GetLeafIndex(vector<double> &ins)
{
	// traverse tree
	int pid = 0;
	TreeNode *curNode = (*this)[pid];
	while (!curNode->isLeaf())
	{
		int fid = curNode->featureId;
		pid = curNode->GetNext(ins[fid]);
		curNode = (*this)[pid];
	}
	return pid;

}

/**
 * @brief: get the leaf index given a sparse instance
 */
int RegTree::GetLeafIdSparseInstance(vector<double> &ins, map<int, int> &fidToDensePos)
{
	int pid = 0; //node id
	TreeNode *curNode = (*this)[pid];
	while (!curNode->isLeaf())
	{
		int fid = curNode->featureId;
		int pos = fidToDensePos[fid];

		if(pos < ins.size())//feature value is available in the dense vector
			pid = curNode->GetNext(ins[pos]);
		else//feature value is stored in the dense vector (due to truncating)
			pid = curNode->GetNext(0);
		curNode = (*this)[pid];
	}

	return pid;
}
