/*
 * commandLineParser.cpp
 *
 *  Created on: 08/12/2014
 *      Author: Zeyi Wen
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "commandLineParser.h"

int Parser::depth = 1;
int Parser::numTree = 1;
float Parser::gamma = 1;
int Parser::numFeature = 0;

void print_null(const char *s) {}

/**
 * @brief: parse a line from terminal
 */
void Parser::ParseLine(int argc, char **argv, char *pcFileName, char *pcSavedFileName)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			HelpInfo();
		switch(argv[i-1][1])
		{
			case 'g':
				gamma = atof(argv[i]);
				break;
			case 'd':
				depth = atoi(argv[i]);
				break;
			case 'n':
				numTree = atoi(argv[i]);
				break;
			case 'f':
				numFeature = atoi(argv[i]);
				if(numFeature < 1)
				{
					HelpInfo();
				}
				break;

			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				HelpInfo();
		}
	}

//	svm_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc)
		HelpInfo();

	strcpy(pcFileName, argv[i]);

	if(i<argc-1)
		strcpy(pcSavedFileName,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(pcSavedFileName,"%s.model",p);
	}
}


void Parser::HelpInfo()
{
	printf(
	"Usage: gbdt xx training_set_file \n"
	"options:\n"
	"-d: depth of the tree\n"
	"-g gamma : set gamma in kernel function\n"
	);
	exit(1);
}


