/*
 * commandLineParser.cpp
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "GBDTCmdLineParser.h"
#include "../Device/CSR/CsrCompressor.h"

int GBDTCmdLineParser::depth = 1;
int GBDTCmdLineParser::numTree = 1;
float GBDTCmdLineParser::gamma = 1;
int GBDTCmdLineParser::numBag = 1;

void print_null(const char *s) {}

/**
 * @brief: parse a line from terminal
 */
bool GBDTCmdLineParser::HandleOption(char c, char *pcOptionValue)
{
	switch(c)
	{
	case 'g':
		gamma = atof(pcOptionValue);
		return true;
	case 'd':
		depth = atoi(pcOptionValue);
		return true;
	case 'n':
		numTree = atoi(pcOptionValue);
		return true;
	case 'b':
		numBag = atoi(pcOptionValue);
		return true;
	case 'c':
		if(atoi(pcOptionValue) == 1)
			CsrCompressor::bUseRle = true;
		return true;
	default:
		return false;
	}
}


void GBDTCmdLineParser::HelpInfo()
{
	printf(
	"Usage: gbdt -d xx -g xx -n xx -b xx training_data_file \n"
	"options:\n"
	"-b: number of bags\n"
	"-n: number of trees"
	"-d: depth of the tree\n"
	"-g: set gamma for regularisation\n"
	);
	exit(1);
}


