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
#include "../../DeviceHost/DefineConst.h"

using std::ostream;

/**
 * @brief: a structure to store split points
 */
class SplitPoint{
public:
	float_point m_fGain;
	float_point m_fSplitValue;
	int m_nFeatureId;

	SplitPoint()
	{
		m_fGain = 0;
		m_fSplitValue = 0;
		m_nFeatureId = -1;
	}

	/**
	 * @brief: return true if values are updated; otherwise false.
	 */
	bool UpdateSplitPoint(float_point fGain, float_point fSplitValue, int nFeatureId)
	{
		if(fGain > m_fGain )//|| (fGain == m_fGain && nFeatureId == m_nFeatureId) NOT USE (second condition is for updating to a new split value)
		{
			m_fGain = fGain;
			m_fSplitValue = fSplitValue;
			m_nFeatureId = nFeatureId;
			return true;
		}
		return false;
	}

	friend ostream& operator << (ostream &os, const SplitPoint &p);
};


#endif /* SPLITPOINT_H_ */
