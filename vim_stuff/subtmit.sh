#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20G

srun hostname
srun sleep 60
