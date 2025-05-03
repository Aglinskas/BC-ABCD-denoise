#!/bin/bash
#SBATCH --job-name=Forrest-PP-BrianIAK
#SBATCH --output=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/Brainiak-papermill-outputs-_%a.txt
#SBATCH --error=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/Brainiak-papermill-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=32gb
#SBATCH --partition=short

notebook_name='132-DeepCorr-BrainIAK-ForStefano-10-6.ipynb'

analysis_name='BrainIAK-CVAE-1'
ofdir='../Data/StudyForrest/ensembles_last_CVAE/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeepCor-ABCD-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s 0 -p r 1 -p analysis_name $analysis_name




