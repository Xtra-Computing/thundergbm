/*
 * DenseInstance.cpp
 *
 *  Created on: 12 Mar 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include <assert.h>
#include <algorithm>
#include <iostream>
#include <set>

#include "DenseInstance.h"

using std::set;
using std::sort;
using std::cout;
using std::endl;
using std::make_pair;

/**
 * @brief: initialise dense instance converter
 */
void DenseInsConverter::InitDenseInsConverter(const vector<RegTree>& vTree)
{
	GetFeatures(vTree);
	GetFidToDensePos();
}

/**
 * @brief: construct a dense instance from a sparse instance
 */
void DenseInsConverter::SparseToDense(const vector<key_value> &sparseIns, vector<double> &denseIns)
{
	int denseInsSize = usedFeaSet.size();
	int sparseInsSize = sparseIns.size();

	//for each value in the sparse instance
	int curDenseTop = 0;
	for(int i = 0; i < sparseInsSize; i++)
	{
		int feaId = sparseIns[i].id;

		while(feaId > usedFeaSet[curDenseTop])
		{
			PushDenseIns(denseIns, curDenseTop, 0);
		}

		if(feaId == usedFeaSet[curDenseTop])
		{//this is a feature needed to be stored in dense instance
			PushDenseIns(denseIns, curDenseTop, sparseIns[i].featureValue);
		}
		else
		{
			//skip this feature, since this feature is not used.
			assert(feaId < usedFeaSet[curDenseTop]);
		}
	}
}

/**
 * @brief: push a value to the dense instance
 */
void DenseInsConverter::PushDenseIns(vector<double> &denseIns, int &curDenseTop, double value)
{
	denseIns.push_back(value);
	curDenseTop++;
}


/**
 * @brief: get the position in the dense instance, given a feature id
 */
int DenseInsConverter::GetPosofDenseIns(int fid)
{
	return fidToDensePos[fid];
}

/**
 * @brief: get features from a set of trees
 */
void DenseInsConverter::GetFeatures(const vector<RegTree>& vTree)
{
	int numofTree = vTree.size();
	set<int> fidSet;

	//get all the used feature ids from trees
	for(int i = 0; i < numofTree; i++)
	{
		//for each node of the tree
		int numofNode = vTree[i].nodes.size();
		for(int j = 0; j < numofNode; j++)
		{
			//skip all the leaves
			if(vTree[i].nodes[j]->isLeaf() == true)
				continue;
			int fid = vTree[i].nodes[j]->featureId;
			fidSet.insert(fid);
		}
	}

//	cout << "number of trees is " << numofTree << "; size of fea set " << fidSet.size() << endl;
	for(set<int>::iterator it = fidSet.begin(); it != fidSet.end(); it++)
	{
		usedFeaSet.push_back(*it);
	}

	//sort the vector ascendantly
	sort(usedFeaSet.begin(), usedFeaSet.end());
}

/**
 * @brief: get a map for mapping a feature id to a dense instance position id
 */
void DenseInsConverter::GetFidToDensePos()
{
	for(int i = 0; i < usedFeaSet.size(); i++)
	{
		fidToDensePos.insert(make_pair(usedFeaSet[i], i));
	}
}


