#!/bin/bash
#SBATCH --job-name=Corr2Reg-baselines
#SBATCH --output=../Data/060-Corr2Reg-baseline/slurm/Corr2Reg-baselines-outputs-_%a.txt
#SBATCH --error=../Data/060-Corr2Reg-baseline/slurm/Corr2Reg-baselines-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=64gb
#SBATCH --array=0-34


#rm ../Data/060-Corr2Reg-baseline/slurm/*.txt
#rm ../Data/060-Corr2Reg-baseline/papermill/*.ipynb

notebook_name='119-make-corr2reg-baselines.ipynb'
outdir=../Data/060-Corr2Reg-baseline/papermill
#fileversion='preproc'
fileversion='deepcor-FG'

datetime
outname=${outdir}/pp-S${SLURM_ARRAY_TASK_ID}-R1-$fileversion.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p fileversion $fileversion

datetime
outname=${outdir}/pp-S${SLURM_ARRAY_TASK_ID}-R2-$fileversion.ipynb
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p fileversion $fileversion





# mkdir -p ${outdir}
# for rep in 0 1 2 3 4
# do

#     outname=${outdir}/075-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R1-rep-$rep.ipynb
#     echo $outname
#     papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p rep $rep -p analysis_name $analysis_name
    
#     outname=${outdir}/075-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R2-rep-$rep.ipynb
#     echo $notebook_name
#     papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p rep $rep -p analysis_name $analysis_name
    
# done

# outname=../Data/040-DeepCor-Papermill/pp-notebooks/072-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
# echo $notebook_name
# papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1

# outname=../Data/040-DeepCor-Papermill/pp-notebooks/072-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
# echo $notebook_name
# papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2




