#!/bin/bash
#SBATCH --job-name=papermill-ABCD-deepcor
#SBATCH --output=../Data/040-DeepCor-Papermill/slurm-files/ABCD-denoise-run-papermill-outputs-_%a.txt
#SBATCH --error=../Data/040-DeepCor-Papermill/slurm-files/ABCD-denoise-run-papermill-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64gb
#SBATCH --partition=gpua100
#SBATCH --array=0-34


notebook_name='076-DeepCorr-pp.ipynb'
outdir=../Data/040-DeepCor-Papermill/pp-notebooks-v2_2_2--linear-newtest-v4
analysis_name='nonlinear-newtest-v4'

mkdir -p ${outdir}
for rep in 0 1 2 3 4
do
    outname=${outdir}/075-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R1-rep-$rep.ipynb
    echo $outname
    papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p rep $rep -p analysis_name $analysis_name
    
    outname=${outdir}/075-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R2-rep-$rep.ipynb
    echo $notebook_name
    papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p rep $rep -p analysis_name $analysis_name
    
done

# outname=../Data/040-DeepCor-Papermill/pp-notebooks/072-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
# echo $notebook_name
# papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1

# outname=../Data/040-DeepCor-Papermill/pp-notebooks/072-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
# echo $notebook_name
# papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2




