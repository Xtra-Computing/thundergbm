/*
 * TreeNode.cpp
 *
 *  Created on: 15 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */


#include "TreeNode.h"

bool TreeNode::isLeaf()
{
	if(featureId == -1)
		return true;
	else
		return false;
}

/*! \brief get next position of the tree given current pid */
int TreeNode::GetNext(float feaValue)
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
