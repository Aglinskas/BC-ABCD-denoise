#!/bin/bash
#SBATCH --job-name=things-papermill-things-deepcor
#SBATCH --output=../Data/things/slurm_files/things-denoise-run-papermill-outputs-v4_%a.txt
#SBATCH --error=../Data/things/slurm_files/things-denoise-run-papermill-errors-v4_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=88gb
#SBATCH --partition=medium
#SBATCH --array=0-17
#SBATCH --nodelist=g[001-019]

# notebook_name='161-refactored-v3-THINGS-loc-v4-ScrubRegress.ipynb'
# analysis_name='refactored-THINGS-loc-v6-fixed-ScrubRegress'

# notebook_name='162-refac-baseline-3-1-THINGS-loc'
# analysis_name='162-refac-baseline-3-1-THINGS-loc-nopreclean'

# notebook_name='162-refac-baseline-3-2-THINGS-loc-TC'
# analysis_name='162-refac-baseline-1-THINGS-TC-nopreclean'

# notebook_name='162-refac-baseline-3-2-THINGS-loc-TC'
# analysis_name='162-refac-baseline-1-THINGS-TC-nopreclean'

# notebook_name='162-refac-baseline-3-1-THINGS-loc'
# analysis_name='162-refac-baseline-1-THINGS-TC-beta-1e-5'

notebook_name='161-refactored-v3-THINGS-loc'
analysis_name='DeepCor-things-nopreclean-beta-1e-5'


ofdir='../Data/StudyForrest/ensembles_last_CVAE/'${analysis_name}
#ofdir='../Data/things/deepcor_outputs/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeepCor-things-idx-${SLURM_ARRAY_TASK_ID}
echo $outname

papermill $notebook_name.ipynb $outname.ipynb --autosave-cell-every 60 --progress-bar -p idx ${SLURM_ARRAY_TASK_ID} -p analysis_name $analysis_name

#rm -rf ../Data/StudyForrest/ensembles_last_CVAE/slurm_files/*


