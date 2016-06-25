/*
 * Predictor.h
 *
 *  Created on: 7 Jan 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include <vector>
#include "Tree/RegTree.h"
#include "KeyValue.h"
#include "../DeviceHost/BaseClasses/BasePredictor.h"

using std::vector;

class HostPredictor: public BasePredictor
{
public:
	void HelloWorld(){}

	/*
public:
	void PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &v_Tree,
					   	  vector<double> &v_fPredValue, vector<double> &v_predBuffer);
	void PredictSparseIns(vector<vector<KeyValue> > &v_vInstance, vector<RegTree> &vTree, vector<double> &v_fPredValue);
	*/
};



#endif /* PREDICTOR_H_ */
