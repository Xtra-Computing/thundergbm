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
 *
 */
void DenseInsConverter::SparseToDense(const vector<KeyValue> &sparseIns, vector<real> &denseIns)
{
	//Caution! @denseIns: can be empty when sparseIns has no features in usedFeaSet

	int denseInsSize = usedFeaSet.size();
	//the tree only has one node.
	if(denseInsSize == 0)
		return;

	int sparseInsSize = sparseIns.size();

	//for each value in the sparse instance
	int curDenseTop = 0;
	for(int i = 0; i < sparseInsSize; i++)
	{
		int feaId = sparseIns[i].id;

		assert(denseInsSize > 0);

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
void DenseInsConverter::PushDenseIns(vector<real> &denseIns, int &curDenseTop, real value)
{
	denseIns.push_back(value);
	curDenseTop++;
}


/**
 * @brief: get the position in the dense instance, given a feature id
 */
int DenseInsConverter::GetPosofDenseIns(int fid)
{
	assert(fidToDensePos.find(fid) != fidToDensePos.end());
	return fidToDensePos[fid];
}

/**
 * @brief: get features from a set of trees
 */
void DenseInsConverter::GetFeatures(const vector<RegTree>& vTree)
{
	//This function has been tested using PrintFeatureVector() @18 Apr 2016

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
	cout << "the numof used features is " << usedFeaSet.size() << endl;

	//sort the vector ascendantly
	sort(usedFeaSet.begin(), usedFeaSet.end());

}

/**
 * @brief: get a map for mapping a feature id to a dense instance position id
 */
void DenseInsConverter::GetFidToDensePos()
{
//	cout << "used features are ";
	for(int i = 0; i < usedFeaSet.size(); i++)
	{
//		cout << usedFeaSet[i] << "\t";
		fidToDensePos.insert(make_pair(usedFeaSet[i], i));
	}
//	cout << endl;
}

/**
 * @brief: print feature set
 */
void DenseInsConverter::PrintFeatureVector()
{
	for(int i = 0; i < usedFeaSet.size(); i++)
	{
		cout << usedFeaSet[i] << "\t";
	}
	cout << endl;
}

void DenseInsConverter::PrintDenseVec(const vector<real> &vDense)
{
	cout << "dense vector: ";
	for(int i = 0; i < vDense.size(); i++)
	{
		cout << vDense[i] << "\t";
	}
	cout << endl;
}

