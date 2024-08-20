#!/bin/bash
#SBATCH -p a6000
#SBATCH -w mp-gpu4-a6000-2
#SBATCH --job-name=dp_train
#SBATCH --output=train.txt
#SBATCH --time=30-00:00:00

python -u train.py
python -u test.py
