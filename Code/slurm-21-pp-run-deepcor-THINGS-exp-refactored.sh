#!/bin/bash
#SBATCH --job-name=papermill-things-deepcor
#SBATCH --output=../Data/things/slurm_files/things-denoise-run-papermill-outputs-v2_%a.txt
#SBATCH --error=../Data/things/slurm_files/things-denoise-run-papermill-errors-v2_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=88gb
#SBATCH --partition=medium
#SBATCH --nodelist=g[001-999]
#SBATCH --array=110-229

notebook_name='161-refactored-v4-THINGS-exp.ipynb'

analysis_name='refactored-THINGS-exp-v1'
ofdir='../Data/StudyForrest/ensembles_last_CVAE/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeepCor-things-idx-${SLURM_ARRAY_TASK_ID}.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p idx ${SLURM_ARRAY_TASK_ID} -p analysis_name $analysis_name

#rm -rf ../Data/StudyForrest/ensembles_last_CVAE/slurm_files/*


