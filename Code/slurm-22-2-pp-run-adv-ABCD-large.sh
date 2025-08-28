#!/bin/bash
#SBATCH --job-name=adv-ABCD-l2
#SBATCH --output=../Data/StudyForrest/DeepCor-slurm/adv-ABCD-v1-outputs-_%a.txt
#SBATCH --error=../Data/StudyForrest/DeepCor-slurm/adv-ABCD-v1-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=32gb
#SBATCH --partition=medium
#SBATCH --array=0-34
#SBATCH --nodelist=g[001-019]


#=0-34
# notebook_name='161-refactored-v2-ABCD'
# analysis_name='refactored-ABCD-v5-nopreclean'

# notebook_name='162-refac-baseline-2-ABCD'
# analysis_name='162-refac-baseline-1-ABCD-nopreclean'

# notebook_name='162-refac-baseline-2-ABCD-TC'
# analysis_name='162-refac-baseline-1-ABCD-TC-nopreclean'

notebook_name='180-refac-2-adversarial-ABCD-large'
analysis_name='neurips-adversarial-ABCD-v2-beta-1e-5'

ofdir='../Data/StudyForrest/ensembles_last_CVAE/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

# outname=${ofdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R1-large.ipynb
# echo $outname
# papermill $notebook_name.ipynb $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p analysis_name $analysis_name

outname=${ofdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R2-large.ipynb
echo $outname
papermill $notebook_name.ipynb $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p analysis_name $analysis_name







