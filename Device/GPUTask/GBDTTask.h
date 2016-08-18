/*
 * VenueRecTask.h
 *
 *  Created on: 14/08/2014
 *      Author: zeyi
 */

#ifndef VENUERECTASK_H_
#define VENUERECTASK_H_

#include "GPUTask.h"

class GBDTTask: public GPUTask
{
public:
	GBDTTask(){}
	virtual int AllocateReusableResources();
	virtual int ReleaseResources();
	virtual void* ProcessTask(void*);
};


#endif /* VENUERECTASK_H_ */
