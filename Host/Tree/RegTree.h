/*
 * RegTree.h
 *
 *  Created on: 7 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef REGTREE_H_
#define REGTREE_H_

#include <map>
#include <vector>
#include "../../DeviceHost/TreeNode.h"
#include "../../SharedUtility/DataType.h"

using std::vector;
using std::map;

class RegTree
{
public:
	vector<TreeNode*> nodes;

public:
	RegTree(){nodes.clear();}

	inline TreeNode* &operator[](int nid)
	{
	    return nodes[nid];
	}

 public:
  int GetLeafIndex(vector<real> &ins);
};



#endif /* REGTREE_H_ */
