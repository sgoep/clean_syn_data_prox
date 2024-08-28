#!/bin/bash
#SBATCH -p a6000
#SBATCH -w mp-gpu4-a6000-3
#SBATCH --job-name=test_op
#SBATCH --output=test_radon_op.txt
#SBATCH --time=30-00:00:00

python -u src/utils/radon_operator.py
