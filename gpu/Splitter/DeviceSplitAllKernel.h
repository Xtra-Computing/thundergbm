/*
 * DeviceSplitAll.h
 *
 *  Created on: 15 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITALL_H_
#define DEVICESPLITALL_H_

#include "../../pureHost/UpdateOps/NodeStat.h"
#include "../../pureHost/UpdateOps/SplitPoint.h"
#include "../../pureHost/BaseClasses/BaseSplitter.h"


typedef double float_point;

__global__ void ComputeWeight(int nNumofKeyValues);



#endif /* DEVICESPLITALL_H_ */
