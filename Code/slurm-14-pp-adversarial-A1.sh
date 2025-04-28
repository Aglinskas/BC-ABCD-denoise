#!/bin/bash
#SBATCH --job-name=papermill-ABCD-deepcor
#SBATCH --output=../Data/StudyForrest/DeepCor-slurm/Forrest-denoise-run-papermill-outputs-_%a.txt
#SBATCH --error=../Data/StudyForrest/DeepCor-slurm/Forrest-denoise-run-papermill-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=32gb
#SBATCH --partition=anzellos
#SBATCH --array=0-13


notebook_name='121-stefano_adversarial-conv-2.ipynb'

analysis_name='new-stefano-conv-allSubs-rep-9'

outdir=../Data/StudyForrest/Stefano_adv_papermill/${analysis_name}
echo $outdir
mkdir -p ${outdir}

outname=${outdir}/ensembling-method-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/ensembling-method-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/ensembling-method-S${SLURM_ARRAY_TASK_ID}-R3.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 3 -p rep 0 -p analysis_name $analysis_name

outname=${outdir}/ensembling-method-S${SLURM_ARRAY_TASK_ID}-R4.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 4 -p rep 0 -p analysis_name $analysis_name



# outname=${outdir}/ensembling-method-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
# if [ ! -f "${outname}" ]; then
#     echo "File ${outname} does not exist running it"
#     papermill $notebook_name $outname --autosave-cell-every 120 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p rep 0 -p analysis_name $analysis_name
# else
#     echo "File ${outname} already exists. Skipping papermill"
# fi

# outname=${outdir}/ensembling-method-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
# if [ ! -f "${outname}" ]; then
#     echo "File ${outname} does not exist running it"
#     papermill $notebook_name $outname --autosave-cell-every 120 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p rep 0 -p analysis_name $analysis_name
# else
#     echo "File ${outname} already exists. Skipping papermill"
# fi

# outname=${outdir}/ensembling-method-S${SLURM_ARRAY_TASK_ID}-R3.ipynb
# if [ ! -f "${outname}" ]; then
#     echo "File ${outname} does not exist running it"
#     papermill $notebook_name $outname --autosave-cell-every 120 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 3 -p rep 0 -p analysis_name $analysis_name
# else
#     echo "File ${outname} already exists. Skipping papermill"
# fi

# outname=${outdir}/ensembling-method-S${SLURM_ARRAY_TASK_ID}-R4.ipynb
# if [ ! -f "${outname}" ]; then
#     echo "File ${outname} does not exist running it"
#     papermill $notebook_name $outname --autosave-cell-every 120 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 4 -p rep 0 -p analysis_name $analysis_name
# else
#     echo "File ${outname} already exists. Skipping papermill"
# fi







