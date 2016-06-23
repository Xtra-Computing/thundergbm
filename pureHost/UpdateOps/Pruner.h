/*
 * Pruner.h
 *
 *  Created on: 2 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef PRUNER_H_
#define PRUNER_H_

#include "../Tree/RegTree.h"

class Pruner
{
public:
	void pruneLeaf(TreeNode** nodes, int nNumofNode);
	static double min_loss;
private:
	int TryPruneLeaf(TreeNode** nodes, int nid, int npruned);
	void ChangeToLeaf(TreeNode* node, double value);

	vector<int> leafChildCnt;
	vector<int> markDelete;
};



#endif /* PRUNER_H_ */
