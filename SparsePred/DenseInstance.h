/*
 * DenseInstance.h
 *
 *  Created on: 12 Mar 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DENSEINSTANCE_H_
#define DENSEINSTANCE_H_

#include <vector>
#include <unordered_map>

#include "../keyValue.h"
#include "../RegTree.h"

using std::vector;
using std::unordered_map;

class DenseInsConverter
{
public:
	vector<int> usedFeaSet;//This vector must be sorted
	unordered_map<int, int> fidToDensePos;//(fid, pid): feature id to position id

public:
	DenseInsConverter(const vector<RegTree>& vTree)
	{
		usedFeaSet.clear();
		fidToDensePos.clear();
		InitDenseInsConverter(vTree);
	}
	void SparseToDense(const vector<key_value> &sparseIns, vector<double> &denseIns);
	int GetPosofDenseIns(int fid);

private:
	void InitDenseInsConverter(const vector<RegTree>& vTree);
	void GetFeatures(const vector<RegTree>& vTree);
	void GetFidToDensePos();
	void PushDenseIns(vector<double> &denseIns, int &curDenseTop, double value);

	//for debugging
public:
	void PrintFeatureVector();
	void PrintDenseVec(const vector<double> &vDense);
};



#endif /* DENSEINSTANCE_H_ */
