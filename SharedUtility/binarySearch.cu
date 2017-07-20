/*
 * binarySearch.cu
 *
 *  Created on: Jun 12, 2017
 *      Author: zeyi
 */
#include <stdio.h>
#include "binarySearch.h"
#include "CudaMacro.h"

__device__ void RangeBinarySearch(uint pos, const uint* pSegStartPos, uint numSeg, uint &segId)
{
	uint midSegId;
	uint startSegId = 0, endSegId = numSeg - 1;
	segId = -1;
	while(startSegId <= endSegId){
		midSegId = startSegId + ((endSegId - startSegId) >> 1);//get the middle index
		CONCHECKER(midSegId < numSeg);
		if(pos >= pSegStartPos[midSegId] && (midSegId == endSegId || pos < pSegStartPos[midSegId + 1]))
		{
			segId = midSegId;
			return;
		}
		else if(pos >= pSegStartPos[midSegId + 1])
			startSegId = midSegId + 1;//find left part
		else{
			ECHECKER(midSegId);
			endSegId = midSegId - 1;//find right part
		}
	}
}
