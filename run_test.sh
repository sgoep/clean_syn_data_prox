#!/bin/bash
#SBATCH -p a6000
#SBATCH -w mp-gpu4-a6000-2
#SBATCH --job-name=dp_test
#SBATCH --output=test.txt
#SBATCH --time=30-00:00:00

python -u test.py