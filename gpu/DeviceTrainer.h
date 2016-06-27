/*
 * DeviceTrainer.h
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICETRAINER_H_
#define DEVICETRAINER_H_

#include "../DeviceHost/BaseClasses/BaseTrainer.h"
#include "Splitter/DeviceSplitter.h"

class DeviceTrainer: public BaseTrainer
{
public:
	DeviceTrainer(DeviceSplitter *pSplitter):BaseTrainer(pSplitter){}
	virtual ~DeviceTrainer(){}

	virtual void InitTree(RegTree &tree);
	virtual void GrowTree(RegTree &tree);
	virtual void ReleaseTree(vector<RegTree> &v_Tree);
};


#endif /* DEVICETRAINER_H_ */
