#!/bin/bash
#SBATCH --job-name=adv-make-ens
#SBATCH --output=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/ABCD-make-ensembled-outputs_%a.txt
#SBATCH --error=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/ABCD-make-ensembled-errors_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=128gb
#SBATCH --partition=medium
#SBATCH --array=0-5


python 141-combine-CVAE-repetitions-ABCD.py


