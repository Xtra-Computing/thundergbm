/*
 * commandLineParser.h
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#ifndef GBDTCOMMANDLINEPARSER_H_
#define GBDTCOMMANDLINEPARSER_H_

#include "../SharedUtility/cmdLineParser.h"

class GBDTCmdLineParser:public CmdLineParser
{
public:
	static int depth;
	static float gamma;
	static int numTree;
	static int numBag;

public:
	GBDTCmdLineParser(){}
	virtual ~GBDTCmdLineParser(){}
	virtual bool HandleOption(char c, char *pcOptionValue);
	virtual void HelpInfo();
};


#endif /* GBDTCOMMANDLINEPARSER_H_ */
