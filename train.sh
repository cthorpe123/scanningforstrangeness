#!/bin/bash
cd /gluster/home/cthorpe/scanningforstrangeness/
source /gluster/home/cthorpe/bin/CondaSetup.sh python3.8DL 
#source /gluster/home/niclane/bin/conda_setup.sh pythondl
pwd
source setup.sh
rm class_weights.npy
#python3 -u train.py -c cfg/lambda.cfg
python3 -u train.py -c cfg/lambda_w_background.cfg
#python3 -u train.py -c cfg/kplus.cfg
