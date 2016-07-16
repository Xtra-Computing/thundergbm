/*
 * Preparator.cpp
 *
 *  Created on: 11 May 2016
 *      Author: Zeyi Wen
 *		@brief: 
 */

#include "Preparator.h"
#include "Memory/gbdtGPUMemManager.h"
#include "Memory/SNMemManager.h"
#include "../DeviceHost/MyAssert.h"
#include "Splitter/DeviceSplitter.h"
#include "Hashing.h"

int *DataPreparator::m_pSNIdToBuffIdHost = NULL;
int *DataPreparator::m_pUsedFIDMap = NULL;




