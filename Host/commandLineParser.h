/*
 * commandLineParser.h
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#ifndef COMMANDLINEPARSER_H_
#define COMMANDLINEPARSER_H_


class Parser
{
public:
	static int depth;
	static float gamma;
	static int numTree;
	static int numFeature;
	static int numBag;

public:
	static void ParseLine(int argc, char **argv, char *pcFileName, char *pcSavedFileName);
	static void HelpInfo();
};


#endif /* COMMANDLINEPARSER_H_ */
