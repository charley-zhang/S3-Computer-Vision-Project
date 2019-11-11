#!/bin/csh 
#$ -M yzhang46@nd.edu
#$ -m bes

#$ -q gpu
#$ -l gpu_card=1

#$ -N CV_Project_Experiments

module load python/3.7.3
module load pytorch
module load cuda

python train.py

