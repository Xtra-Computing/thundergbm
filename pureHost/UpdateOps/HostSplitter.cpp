/*
 * Splitter.cpp
 *
 *  Created on: 12 Apr 2016
 *      Author: Zeyi Wen
 *		@brief: split a node of a tree
 */

#include <algorithm>
#include <math.h>
#include <map>
#include <iostream>

#include "HostSplitter.h"
#include "SplitPoint.h"
#include "../MyAssert.h"

using std::map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

/**
 * @brief: efficient best feature finder
 */
void HostSplitter::FeaFinderAllNode(vector<SplitPoint> &vBest, vector<nodeStat> &rchildStat, vector<nodeStat> &lchildStat)
{
	const float rt_2eps = 2.0 * rt_eps;
	double min_child_weight = 1.0;//follow xgboost

	int nNumofFeature = m_vvFeaInxPair.size();
	vector<nodeStat> tempStat;
	vector<double> vLastValue;
	vector<SplitPoint> vBest16;
	int bufferSize = mapNodeIdToBufferPos.size();

	for(int f = 0; f < nNumofFeature; f++)
	{
		vector<KeyValue> &featureKeyValues = m_vvFeaInxPair[f];
		if(m_nCurDept == 4 && m_nRound == 28 && (f == 15 || f == 46))
		{
			vBest16.clear();
			vBest16.resize(bufferSize);
		}

		int nNumofKeyValues = featureKeyValues.size();

		tempStat.clear();
		vLastValue.clear();
		tempStat.resize(bufferSize);
		vLastValue.resize(bufferSize);

	    for(int i = 0; i < nNumofKeyValues; i++)
	    {
	    	int insId = featureKeyValues[i].id;
			int nid = m_nodeIds[insId];
			PROCESS_ERROR(nid >= -1);
			if(nid == -1)
				continue;

			// start working
			double fvalue = featureKeyValues[i].featureValue;

			// get the statistics of nid node
			// test if first hit, this is fine, because we set 0 during init
			map<int, int>::iterator it = mapNodeIdToBufferPos.find(nid);
			PROCESS_ERROR(it != mapNodeIdToBufferPos.end());
			int bufferPos = it->second;
			if(tempStat[bufferPos].IsEmpty())
			{
				tempStat[bufferPos].Add(m_vGDPair_fixedPos[insId].grad, m_vGDPair_fixedPos[insId].hess);
				vLastValue[bufferPos] = fvalue;
			}
			else
			{
				// try to find a split
				if(fabs(fvalue - vLastValue[bufferPos]) > rt_2eps &&
				   tempStat[bufferPos].sum_hess >= min_child_weight)
				{
					nodeStat lTempStat;
					PROCESS_ERROR(m_nodeStat.size() > bufferPos);
					lTempStat.Subtract(m_nodeStat[bufferPos], tempStat[bufferPos]);
					if(lTempStat.sum_hess >= min_child_weight)
					{
						double loss_chg = CalGain(m_nodeStat[bufferPos], tempStat[bufferPos], lTempStat);
						double sv = static_cast<float>((fvalue + vLastValue[bufferPos]) * 0.5f);
						bool bUpdated = vBest[bufferPos].UpdateSplitPoint(loss_chg, sv, f);
						if(m_nCurDept == 4 && m_nRound == 28 && (f == 15 || f == 46))
						{
							vBest16[bufferPos].UpdateSplitPoint(loss_chg, sv, f);
						}
						if(bUpdated == true)
						{
							lchildStat[bufferPos] = lTempStat;
							rchildStat[bufferPos] = tempStat[bufferPos];
							//if(f == 12 && nid == 262)
							//	printf("fid=%d; node id=%d; fvalue=%f; last_fvalue=%f; sv=%f\n", f, nid, fvalue, vLastValue[bufferPos], sv);
						}
					}
				}
				//update the statistics
				tempStat[bufferPos].Add(m_vGDPair_fixedPos[insId].grad, m_vGDPair_fixedPos[insId].hess);
				vLastValue[bufferPos] = fvalue;
			}
		}

	    // finish updating all statistics, check if it is possible to include all sum statistics
	    for(map<int, int>::iterator it = mapNodeIdToBufferPos.begin(); it != mapNodeIdToBufferPos.end(); it++)
	    {
	    	const int nid = it->first;
            nodeStat lTempStat;
	        lTempStat.Subtract(m_nodeStat[it->second], tempStat[it->second]);
	        if(lTempStat.sum_hess >= min_child_weight && tempStat[it->second].sum_hess >= min_child_weight)
	        {
//	        	cout << "good" << endl;
	        	double loss_chg = CalGain(m_nodeStat[it->second], tempStat[it->second], lTempStat);
	            const float gap = fabs(vLastValue[it->second]) + rt_eps;
	            const float delta = gap;
	            vBest[it->second].UpdateSplitPoint(loss_chg, vLastValue[it->second] + delta, f);
	        }
	    }


		if(m_nCurDept == 4 && m_nRound == 28 && (f == 15 || f == 46))
		{
			PrintVec(vBest16);
		}
	}
}


