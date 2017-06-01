/*
 * BagBuilder.cpp
 *
 *  Created on: 8 Aug 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include <random>
#include <cstring>
#include "BagBuilder.h"
#include "../../SharedUtility/CudaMacro.h"

void BagBuilder::ContructBag(int numIns, int *weight, int numBag)
{
	PROCESS_ERROR(numBag > 0 && numIns > 0 && weight != NULL);
	//set bags to empty
	memset(weight, 0, numIns * numBag * sizeof(int));

	//if only one bag, all the instances are used with equal weight.
	if(numBag == 1)
	{
		for(int j = 0; j < numIns; j++)
		{
			weight[j] = 1;
		}
		return;
	}

	//initialise each bag
	for(int i = 0; i < numBag; i++)
	{
		srand(i + 10);
		for(int j = 0; j < numIns; j++)
		{
			int randNum = rand()%numIns;
			PROCESS_ERROR(randNum <= numIns);
			weight[randNum + i * numIns] += 1;
		}
	}
}

