#!/bin/bash

#SBATCH --time=600
#SBATCH --account=ml4h-jobs
#SBATCH --output=ig.out

data_root=~/ml4h_data/project1/chest_xray

. ~/jupyter/bin/activate
python3 integrated_gradients.py --data_root $data_root
