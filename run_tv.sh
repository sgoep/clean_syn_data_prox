#!/bin/bash
#SBATCH -p a6000
#SBATCH -w mp-gpu4-a6000-2
#SBATCH --job-name=tv
#SBATCH --output=tv.txt
#SBATCH --time=30-00:00:00

python -u total_variation.py
