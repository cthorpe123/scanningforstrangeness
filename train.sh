#!/bin/bash
cd /gluster/home/niclane/scanningforstrangeness
source /gluster/home/niclane/bin/conda_setup.sh pythondl
source setup.sh
python3 train.py -c cfg/default.cfg
