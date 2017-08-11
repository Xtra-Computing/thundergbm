/*
 * memVector.h
 *
 *  Created on: Aug 11, 2017
 *      Author: zeyi
 */

#ifndef MEMVECTOR_H_
#define MEMVECTOR_H_

#include "DataType.h"
class MemVector{
public:
	void *addr = NULL;
	uint size = 0;
	uint reservedSize = 0;
	//reserve memory for a variable
	void reserveSpace(uint newSize, uint numByteEachValue);
};



#endif /* MEMVECTOR_H_ */
