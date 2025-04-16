#!/bin/bash
#SBATCH --job-name=papermill-ABCD-deepcor
#SBATCH --output=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/ABCD-denoise-run-papermill-outputs-_%a.txt
#SBATCH --error=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/ABCD-denoise-run-papermill-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=32gb
#SBATCH --partition=short
#SBATCH --array=0-10

notebook_name='082-DeepCorr-ABCD-face-02.ipynb'

analysis_name='DeepCor-ABCD-v3'
ofdir='../Data/StudyForrest/ensembles_last_CVAE/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeepCor-ABCD-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p analysis_name $analysis_name

outname=${ofdir}/DeepCor-ABCD-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p analysis_name $analysis_name


#rm -rf ../Data/StudyForrest/ensembles_last_CVAE/slurm_files/*


