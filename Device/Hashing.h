/*
 * Hashing.h
 *
 *  Created on: 22 Jun 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef HASHING_H_
#define HASHING_H_

class Hashing
{
public:
	//has an identical version in device @splitAll
	static int HostAssignHashValue(int *pSNIdToBuffId, int orgin, int maxHashValue, bool &bIsNew);

	/**
	 * @brief: return buffer id given a splittable node id
	 */
	static int HostGetBufferId(int *pSNIdToBuffId, int snid, int m_maxNumofSplittable);
};



#endif /* HASHING_H_ */
