#!/bin/bash
#SBATCH -p a6000
#SBATCH -w mp-gpu4-a6000-2
#SBATCH --job-name=create_data
#SBATCH --output=create_data.txt
#SBATCH --time=30-00:00:00

python -u create_data_admm.py
python -u train.py
python -u test.py