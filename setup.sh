#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

RAW_DIR="/gluster/data/microboone/strangeness/raw"
PROC_DIR="/gluster/data/microboone/strangeness/processed"
WORK_DIR="/gluster/home/niclane/scanningforstrangeness"

alias cdr="cd ${RAW_DIR}"
alias cdp="cd ${PROC_DIR}"
alias cds="cd ${WORK_DIR}"

echo -e "${CYAN}-- Available aliases:${NC}"
alias | grep 'cd'

conda activate pythondl
