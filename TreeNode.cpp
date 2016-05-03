/*
 * TreeNode.cpp
 *
 *  Created on: 15 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */


#include "TreeNode.h"

bool TreeNode::isLeaf() const
{
	if(featureId == -1)
		return true;
	else
		return false;
}

bool TreeNode::isRoot() const
{
	if(level == 0)
		return true;
	else
		return false;
}

/*! \brief get next position of the tree given current pid */
int TreeNode::GetNext(double feaValue)
{
    if (feaValue < fSplitValue)
    {
      return leftChildId;
    }
    else
    {
      return rightChildId;
    }
}
