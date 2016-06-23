/*
 * ErrorChecker.h
 *
 *  Created on: 22 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef ERRORCHECKER_H_
#define ERRORCHECKER_H_

__device__ void ErrorCond(bool bCon, const char* functionName, const char* temp);

__device__ void ErrorChecker(int value, const char* functionName, const char* temp);



#endif /* ERRORCHECKER_H_ */
