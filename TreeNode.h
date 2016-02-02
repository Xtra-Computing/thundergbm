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
	int level;

	union{
		int leftChildId;
		int startId;//data (or gdpair) start id
	};
	union{
		int rightChildId;
		int endId;//data (or gdpair) end id
	};


public:

	TreeNode()
	{
		featureId = -1;
		fSplitValue = -1;
		parentId = -1;
		nodeId = -1;
		level = -1;
		leftChildId = -1;
		rightChildId = -1;
	}


	bool isLeaf() const;
	int GetNext(float feaValue);
};


#endif /* TREENODE_H_ */
