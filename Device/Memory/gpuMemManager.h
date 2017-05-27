/*
 * gpuMemAllocator.h
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: Allocate GPU memory for the whole project
 */

#ifndef GPUMEMALLOCATOR_H_
#define GPUMEMALLOCATOR_H_

#include "../../SharedUtility/DataType.h"


class GPUMemManager
{
public:
	static void MemcpyHostToDeviceAsync(void *pHostSrc, void *pDevDst, long long numofByte, void *pStream);
	static void MemcpyDeviceToHostAsync(void *pDevSrc, void *pHostDst, long long numofByte, void *pStream);
	static void MemcpyDeviceToDeviceAsync(void *pDevSrc, void *pDevDst, long long numofByte, void *pStream);
	static void MemsetAsync(void *pDevSrc, int value, long long numofByte, void *pStream);
};



#endif /* GPUMEMALLOCATOR_H_ */
