/*
 * DeviceSplitter.h
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#ifndef DEVICESPLITTER_H_
#define DEVICESPLITTER_H_

#include <vector>

#include "../../Host/UpdateOps/SplitPoint.h"
#include "../../DeviceHost/BaseClasses/BaseSplitter.h"
#include "../../DeviceHost/NodeStat.h"

using std::vector;


class DeviceSplitter: public BaseSplitter
{
public:
	double total_scan_t;
	double total_com_gain_t;
	double total_fix_gain_t;
	double total_fill_gd_t;
	double total_com_idx_t;
	double total_search_t;
	double total_csr_len_t;

	//measure split
	double total_weight_t;
	double total_create_node_t;
	double total_unique_id_t;
	double total_ins2node_t;
	double total_ins2default_t;
	double total_update_new_splittable_t;


public:
	void FeaFinderAllNode(void *pStream, int bagId);
	void FeaFinderAllNode2(void *pStream, int bagId);
	void SplitAll(int &m_nNumofNode, bool bLastLevel, void *pStream, int bagId);
	virtual void ComputeGD(vector<RegTree> &vTree, vector<vector<KeyValue> > & vvInsSparse,  void *stream, int bagId);
};



#endif /* DEVICESPLITTER_H_ */
