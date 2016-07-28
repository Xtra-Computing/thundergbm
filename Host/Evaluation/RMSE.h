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
#include "../../DeviceHost/DefineConst.h"

using std::vector;

class EvalRMSE
{
public:
  char *Name(void);
  float EvalRow(float_point label, float_point pred);
  float GetFinal(float_point esum, float_point wsum);
  float Eval(const vector<float_point> &preds, float_point *labels, int numofIns);
};


#endif /* RMSE_H_ */
