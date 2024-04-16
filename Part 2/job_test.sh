#!/bin/bash

#SBATCH --time=600
#SBATCH --account=ml4h-jobs
#SBATCH --output=test.out

data_root=~/ml4h_data/project1/chest_xray
model_path=models/model_224.pth

. ~/jupyter/bin/activate
python3 test.py --data_root $data_root --model_path $model_path
