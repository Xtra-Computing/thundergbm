#!/bin/bash
declare -a arr=("abalone"
			 	"cadata"
				"covtype"
			 	"e2006"
				"higgs"
				"insurance"
			 	"log1p"
				"real-sim"
				"susy"
			 	"yp"
				"3d")
for i in "${arr[@]}"
do 
	sh ./run.sh $i
	echo "############################## end of $i"
done
