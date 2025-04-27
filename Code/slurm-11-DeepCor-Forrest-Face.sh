#!/bin/bash
#SBATCH --job-name=Forrest-PP
#SBATCH --output=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/DeepCor-BIG-outputs-_%a.txt
#SBATCH --error=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/DeepCor-BIG-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=32gb
#SBATCH --partition=short
#SBATCH --array=0-14

notebook_name='078-DeepCorr-Forrest-ForStefano-10-8-big.ipynb'

analysis_name='DeepCor-Forrest-face-big-16'
ofdir='../Data/StudyForrest/ensembles_last_CVAE/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeepCor-Forrest-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p analysis_name $analysis_name

outname=${ofdir}/DeepCor-Forrest-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p analysis_name $analysis_name

outname=${ofdir}/DeepCor-Forrest-S${SLURM_ARRAY_TASK_ID}-R3.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 3 -p analysis_name $analysis_name

outname=${ofdir}/DeepCor-Forrest-S${SLURM_ARRAY_TASK_ID}-R4.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 4 -p analysis_name $analysis_name


#rm -rf ../Data/StudyForrest/ensembles_last_CVAE/slurm_files/*


