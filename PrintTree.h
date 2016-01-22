/*
 * PrintTree.h
 *
 *  Created on: 20 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef PRINTTREE_H_
#define PRINTTREE_H_

#include <fstream>
#include "RegTree.h"

using std::ofstream;

class TreePrinter
{
public:
	ofstream m_writeOut;

private:
	vector<TreeNode*> m_nodeStack;

public:
	void PrintTree(const RegTree &tree);

private:
	void WriteNode(const TreeNode &node);

	void WriteInternalNode(const TreeNode *node);
	void WriteLeaf(const TreeNode *node);
	void Spacing(int level);
};



#endif /* PRINTTREE_H_ */
