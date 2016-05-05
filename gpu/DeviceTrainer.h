/*
 * DeviceTrainer.h
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICETRAINER_H_
#define DEVICETRAINER_H_

#include "../pureHost/BaseTrainer.h"

class DeviceTrainer: public BaseTrainer
{
public:
	DeviceTrainer(){}
	virtual ~DeviceTrainer(){}

	virtual void GrowTree(RegTree &tree);
};


#endif /* DEVICETRAINER_H_ */
