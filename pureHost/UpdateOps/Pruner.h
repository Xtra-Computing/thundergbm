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
	void pruneLeaf(RegTree &tree);
	static double min_loss;
private:
	int TryPruneLeaf(RegTree &tree, int nid, int npruned);
	void ChangeToLeaf(RegTree &tree, int nid, double value);

	vector<int> leafChildCnt;
	vector<int> markDelete;
};



#endif /* PRUNER_H_ */
