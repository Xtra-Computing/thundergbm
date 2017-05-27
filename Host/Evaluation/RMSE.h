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
#include "../../SharedUtility/DataType.h"

using std::vector;

class EvalRMSE
{
public:
  char *Name(void);
  float EvalRow(real label, real pred);
  float GetFinal(real esum, real wsum);
  float Eval(const vector<real> &preds, real *labels, int numofIns);
};


#endif /* RMSE_H_ */
