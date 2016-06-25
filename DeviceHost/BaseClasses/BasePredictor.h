/*
 * BasePredictor.h
 *
 *  Created on: 27/05/2016
 *      Author: zeyi
 */

#ifndef BASEPREDICTOR_H_
#define BASEPREDICTOR_H_

#include <vector>
#include "../../pureHost/Tree/RegTree.h"
#include "../../pureHost/KeyValue.h"

using std::vector;

class BasePredictor
{
public:
	void PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &v_Tree,
					   	  vector<double> &v_fPredValue, vector<double> &v_predBuffer);
	void PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<double> &v_fPredValue);
};


#endif /* BASEPREDICTOR_H_ */
