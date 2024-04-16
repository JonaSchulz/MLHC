#!/bin/bash

#SBATCH --time=600
#SBATCH --account=ml4h-jobs
#SBATCH --output=train.out

data_root=~/ml4h_data/project1/chest_xray
epochs=200
run_name=model_224_long

. ~/jupyter/bin/activate
python3 train.py --data_root $data_root --epochs $epochs --run_name $run_name
