/*
 * SplitPoint.h
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: store the best split point
 */

#ifndef SPLITPOINT_H_
#define SPLITPOINT_H_

#include <ostream>
#include "../../SharedUtility/DataType.h"

using std::ostream;

/**
 * @brief: a structure to store split points
 */
class SplitPoint{
public:
	real m_fGain;
	real m_fSplitValue;
	int m_nFeatureId;
	bool m_bDefault2Right;

	SplitPoint()
	{
		m_fGain = 0;
		m_fSplitValue = 0;
		m_nFeatureId = -1;
		m_bDefault2Right = false;
	}

	friend ostream& operator << (ostream &os, const SplitPoint &p);
};


#endif /* SPLITPOINT_H_ */
