#!/bin/bash
#make clean
#make -j

if [ $1 -eq 1 ];then
echo "run covtype"
./bin/thundergbm-train data=/home/xbr/ML_dataset/covtype.libsvm.binary.scale
fi

if [ $1 -eq 9 ];then
echo "run E2006"
./bin/thundergbm-train data=/home/xbr/ML_dataset/E2006.train
fi

if [ $1 -eq 2 ];then
echo "run HIGGS"
./bin/thundergbm-train data=/home/xbr/ML_dataset/higgs
fi

if [ $1 -eq 10 ];then
echo "run log1p.E2006.train"
./bin/thundergbm-train data=/home/xbr/ML_dataset/log1p.E2006.train
fi

if [ $1 -eq 3 ];then
echo "run news20.binary"
./bin/thundergbm-train data=/home/xbr/ML_dataset/news20.binary
fi

if [ $1 -eq 4 ];then
echo "run real-sim"
./bin/thundergbm-train data=/home/xbr/ML_dataset/real-sim
fi

if [ $1 -eq 5 ];then
echo "run SUSY"
./bin/thundergbm-train data=/home/xbr/ML_dataset/SUSY
fi

if [ $1 -eq 6 ];then
echo "run rcv1_test"
./bin/thundergbm-train data=/home/xbr/ML_dataset/rcv1_test.binary
fi

if [ $1 -eq 7 ];then
echo "run epsilon_normalized"
./bin/thundergbm-train data=/home/xbr/ML_dataset/epsilon_normalized
fi

if [ $1 -eq 8 ];then
echo "run url_combined_normalized"
./bin/thundergbm-train data=/home/xbr/ML_dataset/url_combined_normalized
fi

