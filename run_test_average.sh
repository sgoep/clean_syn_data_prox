#!/bin/bash
#SBATCH -p a6000
#SBATCH -w mp-gpu4-a6000-2
#SBATCH --job-name=dp_test_average
#SBATCH --output=test_average.txt
#SBATCH --time=30-00:00:00

python -u test_average.py