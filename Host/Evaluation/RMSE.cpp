/*
 * RMSE.cpp
 *
 *  Created on: 4 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: compute rmse
 */

#include <math.h>
#include <iostream>

#include "RMSE.h"

char *EvalRMSE::Name(void)
{
    return "rmse";
}

float EvalRMSE::EvalRow(float_point label, float_point pred)
{
	float_point diff = label - pred;
    return diff * diff;
}

float EvalRMSE::GetFinal(float_point esum, float_point wsum)
{
    return sqrt(esum / wsum);
}

float EvalRMSE::Eval(const vector<float_point> &preds, float_point *labels, int numofIns)
{
	float_point sum = 0.0, wsum = 0.0;
	int ndata = numofIns;
	for (int i = 0; i < ndata; ++i)
	{
		sum += EvalRow(labels[i], preds[i]);
		wsum += 1;
	}
	float_point dat[2]; dat[0] = sum, dat[1] = wsum;
	return GetFinal(dat[0], dat[1]);
}

