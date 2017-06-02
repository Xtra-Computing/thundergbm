/*
 * BaseSplitter.cpp
 *
 *  Created on: 5 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */


#include <algorithm>
#include <math.h>
#include <map>
#include <unordered_map>
#include <iostream>

#include "../../Host/UpdateOps/SplitPoint.h"
#include "../../SharedUtility/CudaMacro.h"
#include "BaseSplitter.h"

using std::unordered_map;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

vector<vector<KeyValue> > BaseSplitter::m_vvFeaInxPair; //value is feature value (sorted in a descendant order); id (or key) is instance id
unordered_map<int, int> BaseSplitter::mapNodeIdToBufferPos;
vector<int> BaseSplitter::m_nodeIds; //instance id to node id
vector<gdpair> BaseSplitter::m_vGDPair_fixedPos;
vector<nodeStat> BaseSplitter::m_nodeStat; //all the constructed tree nodes
real BaseSplitter::m_lambda;//the weight of the cost of complexity of a tree
real BaseSplitter::m_gamma;//the weight of the cost of the number of trees


