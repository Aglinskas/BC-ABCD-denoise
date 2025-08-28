#!/bin/bash
#SBATCH --job-name=adv-things
#SBATCH --output=../Data/things/slurm_files/adv-things-v3-outputs-v4_%a.txt
#SBATCH --error=../Data/things/slurm_files/adv-things-v3-errors-v4_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=32gb
#SBATCH --partition=medium
#SBATCH --array=0-17
#SBATCH --nodelist=g[001-019]

notebook_name='180-refac-3-adversarial-THINGS'
analysis_name='neurips-adversarial-things-v3-beta-1e-5'

ofdir='../Data/StudyForrest/ensembles_last_CVAE/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeepCor-things-idx-${SLURM_ARRAY_TASK_ID}
echo $outname

papermill $notebook_name.ipynb $outname.ipynb --autosave-cell-every 60 --progress-bar -p idx ${SLURM_ARRAY_TASK_ID} -p analysis_name $analysis_name

#rm -rf ../Data/StudyForrest/ensembles_last_CVAE/slurm_files/*


