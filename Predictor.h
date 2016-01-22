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
#include "RegTree.h"

using std::vector;

class Predictor
{
public:
	void Predict(vector<vector<double> > &v_vInstance, vector<RegTree> &v_Tree,
				 vector<double> &v_fPredValue, vector<double> &v_predBuffer);
};



#endif /* PREDICTOR_H_ */
