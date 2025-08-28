#!/bin/bash
#SBATCH --job-name=adv-forrest
#SBATCH --output=../Data/StudyForrest/DeepCor-slurm/papermill-adv-forrest-v1-outputs-_%a.txt
#SBATCH --error=../Data/StudyForrest/DeepCor-slurm/papermill-adv-forrest-v1-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64gb
#SBATCH --partition=medium
#SBATCH --array=0-14
#SBATCH --nodelist=g[001-019]


#notebook_name='161-refactored-v1-forrest'
#notebook_name='162-refac-baseline-1-forrest'

notebook_name='180-refac-1-adversarial-forrest'
analysis_name='neurips-adversarial-forrest-v1-beta-1e-5'

outdir=../Data/StudyForrest/ensembles_last_CVAE/${analysis_name}
echo $outdir
mkdir -p ${outdir}

outname=${outdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
papermill $notebook_name.ipynb $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
papermill $notebook_name.ipynb $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R3.ipynb
papermill $notebook_name.ipynb $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 3 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/$notebook_name-S${SLURM_ARRAY_TASK_ID}-R4.ipynb
papermill $notebook_name.ipynb $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 4 -p rep 0 -p analysis_name $analysis_name



