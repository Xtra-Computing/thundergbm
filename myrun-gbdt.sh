#!/bin/bash
declare -a arr=("abalone"
			 	"cadata"
			 	"e2006"
			 	"log1p"
			 	"yp")
for i in "${arr[@]}"
do 
	sh ./run.sh $i
	echo "############################## end of $i"
done
