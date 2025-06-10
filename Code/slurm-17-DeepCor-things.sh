#!/bin/bash
#SBATCH --job-name=papermill-things-deepcor
#SBATCH --output=../Data/things/slurm_files/things-denoise-run-papermill-outputs-v2_%a.txt
#SBATCH --error=../Data/things/slurm_files/things-denoise-run-papermill-errors-v2_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=11:00:00
#SBATCH --mem=88gb
#SBATCH --partition=short
#SBATCH --array=0-17

notebook_name='151-DeepCorr-things-face-01.ipynb'

analysis_name='DeepCor-things-v1'
ofdir='../Data/things/deepcor_outputs/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeepCor-things-idx-${SLURM_ARRAY_TASK_ID}.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p idx ${SLURM_ARRAY_TASK_ID} -p analysis_name $analysis_name

#rm -rf ../Data/StudyForrest/ensembles_last_CVAE/slurm_files/*


