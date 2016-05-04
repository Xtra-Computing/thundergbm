/*
 * SplitPoint.cpp
 *
 *  Created on: 27 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: define functions for split points
 */

#include <iostream>
#include "SplitPoint.h"

using std::endl;

ostream& operator << (ostream &os, const SplitPoint &p)
{
	os << "Gain=" << p.m_fGain << "; sv=" << p.m_fSplitValue << "; fid=" << p.m_nFeatureId << "\t";
	return os;
}
