#!/bin/bash
declare -a arr=(
				"abalone"
			 	"cadata"
				"covtype"
			 	"e2006"
				"higgs"
				"insurance"
			 	"log1p"
				"new20"
				"real-sim"
				"susy"
			 	"yp"
				"3d")

for i in "${arr[@]}"
do 
	if [ "$1" == "l" ]
	then	sh ./run.sh $i
	else	sh ./small-run.sh $i
	fi
	echo "############################## end of $i"
done
