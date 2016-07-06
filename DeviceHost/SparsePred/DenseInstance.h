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
#include <map>

#include "../../Host/KeyValue.h"
#include "../../Host/Tree/RegTree.h"

using std::vector;
using std::map;

class DenseInsConverter
{
public:
	vector<int> usedFeaSet;//This vector must be sorted
	map<int, int> fidToDensePos;//(fid, pid): feature id to position id

public:
	DenseInsConverter(const vector<RegTree>& vTree)
	{
		usedFeaSet.clear();
		fidToDensePos.clear();
		InitDenseInsConverter(vTree);
	}
	void SparseToDense(const vector<KeyValue> &sparseIns, vector<float_point> &denseIns);
	int GetPosofDenseIns(int fid);

private:
	void InitDenseInsConverter(const vector<RegTree>& vTree);
	void GetFeatures(const vector<RegTree>& vTree);
	void GetFidToDensePos();
	void PushDenseIns(vector<float_point> &denseIns, int &curDenseTop, float_point value);

	//for debugging
public:
	void PrintFeatureVector();
	void PrintDenseVec(const vector<float_point> &vDense);
};



#endif /* DENSEINSTANCE_H_ */
