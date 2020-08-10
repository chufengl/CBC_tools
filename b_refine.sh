#!/bin/bash

#SBATCH --job-name=DE_ref
#SBATCH -p upex
##SBATCH --mem-per-cpu=8000
#SBATCH -n 1
##SBATCH -N 1
#SBATCH -t 2-12:00    
##SBATCH --array=20%8

##SBATCH -A chufengl
#SBATCH -o DEREF_%j.out
#SBATCH -e DEREF_%j.err
##SBATCH --mail-type=ALL
##SBATCH --mail-type=END        # notifications for job done & fail
#SBATCH --mail-user=chufeng.li@cfel.de # send-to address

#source ~/anaconda3/bin/activate base

export PYTHONUNBUFFERED=1

PYTHON=~/anaconda3/bin/python

start_frame=$1
end_frame=$2

round=10

mkdir fr${start_frame}_${end_frame}
cd fr${start_frame}_${end_frame}
for  ((r=1;r<=$round;r++));
do 
	mkdir round$r
	cd round$r
	export PYTHONUNBUFFERED=1
	$PYTHON -u /home/lichufen/CCB_ind/scripts/batch_refine.py $start_frame $end_frame
	cd ..
done


