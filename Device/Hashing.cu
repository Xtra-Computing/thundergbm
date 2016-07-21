/*
 * Hashing.cpp
 *
 *  Created on: 22 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "Hashing.h"
#include "../DeviceHost/MyAssert.h"
#include "DeviceHashing.h"

/**
 * @brief: a hash function (has an identical version in device @splitAll)
 * @bIsNew: for checking if the hash value is newly produced.
 *
 */
int Hashing::HostAssignHashValue(int *pEntryToHashValue, int origin, int maxHashValue, bool &bIsNew)
{
	int buffId = AssignHashValue(pEntryToHashValue, origin, maxHashValue, bIsNew);
	PROCESS_ERROR(buffId > -1);
	return buffId;
}

/**
 * @brief: has an identical verion in device
 */
int Hashing::HostGetBufferId(const int *pSNIdToBuffId, int snid, int m_maxNumofSplittable)
{
	int buffId = GetBufferId(pSNIdToBuffId, snid, m_maxNumofSplittable);

	return buffId;
}
