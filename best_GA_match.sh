#!/bin/bash

#SBATCH --job-name=Best_GA_match
#SBATCH -p upex
##SBATCH --mem-per-cpu=8000
#SBATCH -n 1
##SBATCH -N 1
#SBATCH -t 2-12:00    
##SBATCH --array=20%8

##SBATCH -A chufengl
#SBATCH -o BGA_match_%j.out
#SBATCH -e BGA_match_%j.err
##SBATCH --mail-type=ALL
#SBATCH --mail-type=END        # notifications for job done & fail
##SBATCH --mail-user=chufeng.li@cfel.de # send-to address

source ~/anaconda3/bin/activate base

export PYTHONUNBUFFERED=1

PYTHON=~/anaconda3/bin/python

exp_img_file=$1
res_file=$2

export PYTHONUNBUFFERED=1
$PYTHON -u /home/lichufen/CCB_ind/scripts/gen_match_figs.py $exp_img_file $res_file


