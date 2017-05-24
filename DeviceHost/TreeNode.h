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

	unsigned int numIns;
	bool m_bDefault2Right;	//instances with missing values go to left node by default

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
		base_weight = -1;
		numIns = 0;
		m_bDefault2Right = false;
	}


	bool isLeaf() const;
	bool isRoot() const;
	int GetNext(double feaValue);
};


#endif /* TREENODE_H_ */
