#!/bin/bash
#SBATCH --job-name=adv-make-ens
#SBATCH --output=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/adv-make-ensembled-outputs_%a.txt
#SBATCH --error=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/adv-make-ensembled-errors_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32gb
#SBATCH --partition=medium
#SBATCH --array=0-10


python 120-organize-make-ensembled-ims.py


