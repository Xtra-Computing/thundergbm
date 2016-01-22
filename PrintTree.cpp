/*
 * PrintTree.cpp
 *
 *  Created on: 20 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <iostream>
#include <iomanip>

#include "PrintTree.h"

using std::cout;
using std::endl;

void TreePrinter::Spacing(int level)
{
	for(int i = 0; i < level; i++)
		m_writeOut << "\t";
}

void TreePrinter::WriteInternalNode(const TreeNode *node)
{
	Spacing(node->level);
	m_writeOut << node->nodeId << ":[f" << node->featureId + 1 << "<";
	m_writeOut.precision(5);
	m_writeOut << node->fSplitValue << "]";
	m_writeOut << " yes=" << node->leftChildId << ",no=" << node->rightChildId << ",missing=" << node->leftChildId << "\n";
}

void TreePrinter::WriteLeaf(const TreeNode *node)
{
	Spacing(node->level);
	m_writeOut << node->nodeId << ":leaf=";
	m_writeOut.precision(5);
	m_writeOut << node->predValue << "\n";
}

void TreePrinter::PrintTree(const RegTree &tree)
{
	m_nodeStack.clear();//store id

	TreeNode *node = tree.nodes[0];
	do
	{
		if(node->isLeaf() == false)
		{
			WriteInternalNode(node);
			if(tree.nodes[node->leftChildId]->isLeaf() == true)
			{
				WriteLeaf(tree.nodes[node->leftChildId]);
				WriteLeaf(tree.nodes[node->rightChildId]);
				if(m_nodeStack.empty())
					break;
				node = m_nodeStack.back();
				m_nodeStack.pop_back();
			}
			else
			{
				m_nodeStack.push_back(tree.nodes[node->rightChildId]);
				node = tree.nodes[node->leftChildId];
			}
		}
		else
		{
			WriteLeaf(node);
			if(m_nodeStack.empty())
				break;
		}

	}while(true);
}
