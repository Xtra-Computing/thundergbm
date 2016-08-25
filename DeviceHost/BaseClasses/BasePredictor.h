/*
 * BasePredictor.h
 *
 *  Created on: 27/05/2016
 *      Author: zeyi
 */

#ifndef BASEPREDICTOR_H_
#define BASEPREDICTOR_H_

#include <vector>
#include "../../Host/Tree/RegTree.h"
#include "../../Host/KeyValue.h"

using std::vector;

class BasePredictor
{
public:
	virtual~BasePredictor(){}

	void PredictSparseInsByLastTree(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &v_Tree,
					   	  vector<float_point> &v_fPredValue, vector<float_point> &v_predBuffer);
	virtual void PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<float_point> &v_fPredValue, void *pStream, int bagId) = 0;
};


#endif /* BASEPREDICTOR_H_ */
