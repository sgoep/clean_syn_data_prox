#!/bin/bash
#SBATCH -p a100
#SBATCH -w mp-gpu4-a100-2
#SBATCH --job-name=test_data
#SBATCH --output=test_data.txt
#SBATCH --time=30-00:00:00

python -u test_data.py