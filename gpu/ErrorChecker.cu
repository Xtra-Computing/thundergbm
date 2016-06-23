/*
 * ErrorChecker.cu
 *
 *  Created on: 22 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>
#include "ErrorChecker.h"

__device__ void ErrorCond(bool bCon, const char* functionName, const char* temp)
{
	if(bCon == false)
	{
		printf("Error in %s: %s=%d\n", functionName, temp);
	}
}

__device__ void ErrorChecker(int value, const char* functionName, const char* temp)
{
	if(value < 0)
	{
		printf("Error in %s: %s=%d\n", functionName, temp, value);
	}
}


