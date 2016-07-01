/*
 * DeviceHashing.h
 *
 *  Created on: 22 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICEHASHING_H_
#define DEVICEHASHING_H_

__device__ __host__ int AssignHashValue(int *pEntryToHashValue, int snid, int m_maxNumofSplittable, bool &bIsNew);

__device__ __host__ int GetBufferId(const int *pSNIdToBuffId, int snid, int m_maxNumofSplittable);

#endif /* DEVICEHASHING_H_ */
