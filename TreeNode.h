/*
 * TreeNode.h
 *
 *  Created on: 7 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef TREENODE_H_
#define TREENODE_H_


class TreeNode
{
public:
	int featureId;

	union{
		float fSplitValue;
		float predValue;
	};

	int parentId;
	int nodeId;

	union{
		int leftChildId;
		int startId;
	};
	union{
		int rightChildId;
		int endId;
	};


public:

	TreeNode()
	{
		featureId = -1;
		fSplitValue = -1;
		parentId = -1;
		nodeId = -1;
		leftChildId = -1;
		rightChildId = -1;
	}


	bool isLeaf();
	int GetNext(float feaValue);
};


#endif /* TREENODE_H_ */
