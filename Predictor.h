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
	void Predict(vector<vector<float> > &v_vInstance, vector<RegTree> &v_Tree,
				 vector<float> &v_fPredValue, vector<float> &v_predBuffer);
};



#endif /* PREDICTOR_H_ */
