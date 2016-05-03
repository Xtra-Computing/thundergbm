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
		double fSplitValue;
		double predValue;
	};

	int parentId;
	int nodeId;
	int level;

	int leftChildId;
	int rightChildId;

	double loss;
	double base_weight;

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

		loss = -1.0;
	}


	bool isLeaf() const;
	bool isRoot() const;
	int GetNext(double feaValue);
};


#endif /* TREENODE_H_ */
