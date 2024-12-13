#!/bin/bash
#SBATCH --job-name=papermill-ABCD-deepcor
#SBATCH --output=../Data/StudyForrest/DeepCor-slurm/Forrest-denoise-run-papermill-outputs-_%a.txt
#SBATCH --error=../Data/StudyForrest/DeepCor-slurm/Forrest-denoise-run-papermill-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64gb
#SBATCH --partition=gpua100
#SBATCH --array=0-13


notebook_name='077-DeepCorr-Forrest-LIN-newINIT.ipynb'
analysis_name='forrest-LIN-newInit-overfit-1'

outdir=../Data/StudyForrest/DeepCor-papermill/${analysis_name}
echo $outdir
mkdir -p ${outdir}

outname=${outdir}/077-DeepCorr-Forrest-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/077-DeepCorr-Forrest-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/077-DeepCorr-Forrest-S${SLURM_ARRAY_TASK_ID}-R3.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 3 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/077-DeepCorr-Forrest-S${SLURM_ARRAY_TASK_ID}-R4.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 4 -p rep 0 -p analysis_name $analysis_name






