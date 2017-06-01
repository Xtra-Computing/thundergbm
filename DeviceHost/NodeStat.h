/*
 * NodeStat.h
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: node statistics
 */

#ifndef NODESTAT_H_
#define NODESTAT_H_


#include <iostream>
#include <stdlib.h>
#include "../SharedUtility/DataType.h"

using std::cout;
using std::endl;

class nodeStat{
public:
	real sum_gd;
	real sum_hess;

	nodeStat()
	{
		sum_gd = 0.0;
		sum_hess = 0.0;
	}
};



#endif /* NODESTAT_H_ */
