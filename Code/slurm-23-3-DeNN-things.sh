#!/bin/bash
#SBATCH --job-name=DeNN-things
#SBATCH --output=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/DeNN-things-v1-outputs-_%a.txt
#SBATCH --error=../Data/StudyForrest/ensembles_last_CVAE/slurm_files/DeNN-things-v1-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32gb
#SBATCH --partition=medium
#SBATCH --array=0-17


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/aglinska/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/aglinska/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/aglinska/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/aglinska/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda init bash
conda activate DeNN

notebook_name='192-DeNN-things.ipynb'

analysis_name='DeNN-things-neurips-v1'
ofdir='../Data/StudyForrest/ensembles_last_CVAE/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeNN-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p idx ${SLURM_ARRAY_TASK_ID} -p analysis_name $analysis_name

