/*
 * RMSE.h
 *
 *  Created on: 4 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef RMSE_H_
#define RMSE_H_

#include <vector>

using std::vector;

class EvalRMSE
{
public:
  char *Name(void);
  float EvalRow(float label, float pred);
  float GetFinal(float esum, float wsum);
  float Eval(const vector<double> &preds, const vector<double> &labels);
};


#endif /* RMSE_H_ */
