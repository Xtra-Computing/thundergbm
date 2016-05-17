/*
 * gpuMemAllocator.h
 *
 *  Created on: 4 May 2016
 *      Author: Zeyi Wen
 *		@brief: Allocate GPU memory for the whole project
 */

#ifndef GPUMEMALLOCATOR_H_
#define GPUMEMALLOCATOR_H_


typedef double float_point;


class GPUMemManager
{
public:
	void MemcpyHostToDevice(void *pHostSrc, void *pDevDst, int numofByte);
	void MemcpyDeviceToHost(void *pDevSrc, void *pHostDst, int numofByte);
	void MemcpyDeviceToDevice(void *pDevSrc, void *pDevDst, int numofByte);
	void Memset(void *pDevSrc, int value, int numofByte);

	void TestMemcpyHostToDevice(void *pHostSrc, void *pDevDst, int numofByte);
	void TestMemcpyDeviceToHost();
	void TestMemcpyDeviceToDevice();
};



#endif /* GPUMEMALLOCATOR_H_ */
