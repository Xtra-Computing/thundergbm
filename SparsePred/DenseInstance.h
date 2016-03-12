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

#include "../keyValue.h"
#include "../RegTree.h"

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
		InitDenseInsConverter(vTree);
	}
	void SparseToDense(const vector<key_value> &sparseIns, vector<double> &denseIns);
	int GetPosofDenseIns(int fid);

private:
	void InitDenseInsConverter(const vector<RegTree>& vTree);
	void GetFeatures(const vector<RegTree>& vTree);
	void GetFidToDensePos();
	void PushDenseIns(vector<double> &denseIns, int &curDenseTop, double value);
};



#endif /* DENSEINSTANCE_H_ */
