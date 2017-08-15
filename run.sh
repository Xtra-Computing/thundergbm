#!/usr/bin/env bash
print_usage(){
    printf "usage: ./run.sh [dataset]\n"
    printf "\tsize\tclass\tfeature\n"
    printf "iris\t150\t3\t4\n"
    printf "mnist\t60000\t10\t780\n"
    printf "a9a\t32561\t2\t123\n"
    printf "a6a\t11220\t2\t123\n"
    printf "news20\t19996\t2\t1335191\n"
}
if [ $# != 1 ]
then
    print_usage
    exit
fi
DATASET_DIR=dataset
#number of bags
BAG="-b "
#gamma for regularisation 
GAMMA="-g "
#depth of trees
D="-d 6"
#number of trees
NUMTREE="-n 40"
#use CSR compression: 0 for not using CSR; 1 for using CSR.
CSR="-c 1"
#file name (must appear as the last argument)
case $1 in
	abalone)
		GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
		FILENAME=${DATASET_DIR}/"abalone.txt" #8 features and 4177 instances
		;;
    cadata)
        GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"cadata" #8 features and 20640 instances
        ;;
	covtype)
		GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"covtype" #90 features and 41M instances
		;;
    yp)
	   	GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"YearPredictionMSD" #90 features and 41M instances
        ;;
    e2006)
		GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"E2006.train" #150360 features and 16087 instances
        ;;
	insurance)
		GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"ins.libsvm" #features and 11M instances
        ;;

    log1p)
        GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
	    FILENAME=${DATASET_DIR}/"log1p.E2006.train" #4,272,227 features and 16,087 instances
        ;;
    new20)
        GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
	    FILENAME=${DATASET_DIR}/"news20.binary" #4,272,227 features and 16,087 instances
        ;;
    3d)
        GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"3d_spatial_network.txt"
        ;;
    higgs)
        GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"HIGGS"
        ;;
    real-sim)
		GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"real-sim" #
        ;;
	susy)
		GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"SUSY"
        ;;
	webspam)
		GAMMA=${GAMMA}"1"
		BAG=${BAG}"1"
        FILENAME=${DATASET_DIR}/"webspam_wc_normalized_trigram.svm"
        ;;
    *)
        echo "undefined dataset, use GAMMA=1, D=8"
        GAMMA=${GAMMA}"1"
        FILENAME=${DATASET_DIR}/$1
esac
###options

#task type: 0 for training; 1 for cross-validation; 2 for evaluation
#	    3 for grid search; 4 for selecting better C.
TASK="-o 2"

#number of features
#NUMFEATURE="-f 16"

#print out the command before execution
set -x

#command
./bin/release/gbdt ${GAMMA} ${D} ${NUMTREE} ${CSR} ${BAG} ${FILENAME}
