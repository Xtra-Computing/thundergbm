/*
 * gpuMemAllocator.h
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: Allocate GPU memory for the whole project
 */

#ifndef GPUMEMALLOCATOR_H_
#define GPUMEMALLOCATOR_H_

#include "../../DeviceHost/DefineConst.h"


class GPUMemManager
{
public:
	static void MemcpyHostToDevice(void *pHostSrc, void *pDevDst, int numofByte);
	static void MemcpyDeviceToHost(void *pDevSrc, void *pHostDst, int numofByte);
	static void MemcpyDeviceToDevice(void *pDevSrc, void *pDevDst, int numofByte);
	static void Memset(void *pDevSrc, int value, int numofByte);

	void TestMemcpyHostToDevice(void *pHostSrc, void *pDevDst, int numofByte);
	void TestMemcpyDeviceToHost();
	void TestMemcpyDeviceToDevice();
};



#endif /* GPUMEMALLOCATOR_H_ */
