/*
 * RegTree.cpp
 *
 *  Created on: 15 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "RegTree.h"

  /*!
   * \brief get the leaf index
   * \param feats dense feature vector, if the feature is missing the field is set to NaN
   * \param root_gid starting root index of the instance
   * \return the leaf index of the given feature
   */
int RegTree::GetLeafIndex(vector<float> &ins)
{
	// tranverse tree
	int pid = 0;
	TreeNode &curNode = (*this)[pid];
	while (!curNode.isLeaf())
	{
		int fid = curNode.featureId;
		pid = curNode.GetNext(ins[fid]);
		curNode = (*this)[pid];
	}
	return pid;

}
