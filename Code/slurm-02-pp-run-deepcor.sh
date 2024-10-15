#!/bin/bash
#SBATCH --job-name=papermill-ABCD-deepcor
#SBATCH --output=../Data/040-DeepCor-Papermill/slurm-files/ABCD-denoise-run-papermill-outputs-_%a.txt
#SBATCH --error=../Data/040-DeepCor-Papermill/slurm-files/ABCD-denoise-run-papermill-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --mem=64gb
#SBATCH --partition=gpua100
#SBATCH --array=25-34


notebook_name='072-DeepCorr-gamma.ipynb'

outname=../Data/040-DeepCor-Papermill/pp-notebooks/072-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
echo $notebook_name
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1

outname=../Data/040-DeepCor-Papermill/pp-notebooks/072-DeepCorr-gamma-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
echo $notebook_name
papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2


# echo $0

# bash

# source .bash_profile

# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/data/aglinska/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/data/aglinska/anaconda3/etc/profile.d/conda.sh" ]; then
#         . "/data/aglinska/anaconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/data/aglinska/anaconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<

# conda activate tf231