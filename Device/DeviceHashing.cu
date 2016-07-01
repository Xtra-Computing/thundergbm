/*
 * DeviceHashing.cu
 *
 *  Created on: 22 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <stdio.h>

#include "DeviceHashing.h"

/**
 * @brief: has an identical version in host
 */
__device__ __host__ int AssignHashValue(int *pEntryToHashValue, int origin, int maxHashValue, bool &bIsNew)
{
	bIsNew = false;//
	int buffId = -1;

	int remain = origin % maxHashValue;//use mode operation as Hash function to find the buffer position

	//the entry has been seen before, and is found without hash conflict
	if(pEntryToHashValue[remain] == origin)
	{
		return remain;
	}

	//the entry hasn't been seen before, and its hash value is found without hash conflict
	if(pEntryToHashValue[remain] == -1)
	{
		bIsNew = true;
		buffId = remain;
		pEntryToHashValue[remain] = origin;
	}
	else//the hash value is used for other entry
	{
		//Hash conflict
		for(int i = maxHashValue - 1; i > 0; i--)
		{
			bool hashValueFound = false;
			if(pEntryToHashValue[i] == -1)//the entry hasn't been seen before, and now is assigned a hash value.
			{
				hashValueFound = true;
				bIsNew = true;
			}
			else if(pEntryToHashValue[i] == origin)//the entry has been seen before, and now its hash value is found.
				hashValueFound = true;

			if(hashValueFound == true)
			{
				buffId = i;
				break;
			}
		}
	}

	return buffId;
}

__device__ __host__ int GetBufferId(const int *pEntryToHashValue, int entry, int maxHashValue)
{
	int buffId = -1;

	int remain = entry % maxHashValue;//use mode operation as Hash function to find the buffer position

	//checking where snid is located
	if(remain < 0)
		printf("hash value error: %d\n", remain);
	if(pEntryToHashValue[remain] == entry)
	{
		buffId = remain;
	}
	else
	{
		//Hash conflict
		for(int i = maxHashValue - 1; i > 0; i--)
		{
			if(pEntryToHashValue[i] == entry)
				buffId = i;
		}
	}

	return buffId;
}

