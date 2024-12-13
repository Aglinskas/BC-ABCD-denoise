#!/bin/bash
#SBATCH --job-name=papermill-ABCD-deepcor
#SBATCH --output=papermill-ABCD-deepcor-outputs.txt
#SBATCH --error=papermill-ABCD-deepcor-errors.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64gb
#SBATCH --partition=gpua100


notebook_name='076-DeepCorr-pp-NL-vanilla'
papermill ${notebook_name}.ipynb ${notebook_name}-executed3.ipynb --autosave-cell-every 5 --progress-bar






