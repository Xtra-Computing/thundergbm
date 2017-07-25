/*
 * CsrCompressor.h
 *
 *  Created on: Jul 25, 2017
 *      Author: zeyi
 */

#ifndef CSRCOMPRESSOR_H_
#define CSRCOMPRESSOR_H_

#include "../../SharedUtility/DataType.h"

class CsrCompressor
{
public:
	void CsrCompression(uint &totalNumCsrFvalue, uint *eachCompressedFeaStartPos, uint *eachCompressedFeaLen,
		uint *eachNodeSizeInCsr, uint *eachCsrNodeStartPos);
};



#endif /* CSRCOMPRESSOR_H_ */
