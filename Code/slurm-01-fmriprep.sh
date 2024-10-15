#!/bin/bash
#SBATCH --job-name=ABCD_fMRIprep_nBack
#SBATCH --output=../Data/011-SLURM-fMRIprep-outs/output_fmriprep_%a
#SBATCH --error=../Data/011-SLURM-fMRIprep-outs/error_fmriprep_%a
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=64gb
#SBATCH --array=0-30%10


module load singularity/

date

SLURM_ARRAY_TASK_ID=$(($SLURM_ARRAY_TASK_ID+0))

echo $SLURM_ARRAY_TASK_ID

data_dir=../Data/010-Data_bids2
output_dir=../Data/020-fmriprepped
sing_image=$HOME/fmriprep-22

cd $data_dir
#files=(sub*)

files=(sub*)
#files=('sub-NDARINV3H5M0JJR')
#echo ${files[0]}

cd /data/aglinska/BC-ABCD-denoise/Code

fs_lic=$HOME'/fs_licence.txt'

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub


ofn=$output_dir/$sub # Define output directory

# Checks if OFN exists, if so then skips fMRIprep
if test -d $ofn; then
    echo "Directory exists, skipping"
else
    working_dir=/scratch/aglinska/fmriprep-workingdir/f0_${sub}
    scratch_dir=/scratch/aglinska/fmriprep-scratch/f0_${sub}
    
    ## AA TODO: ADD 'export SINGULARITY_TMPDIR=' to fix rootfs issue
    
    mkdir -p /scratch/aglinska/fmriprep-workingdir
    mkdir -p /scratch/aglinska/fmriprep-scratch
    
    chmod 777 /scratch/aglinska/fmriprep-workingdir
    chmod 777 /scratch/aglinska/fmriprep-scratch
    
    mkdir -p $working_dir
    mkdir -p $scratch_dir
    
    chmod 777 $scratch_dir
    chmod 777 $working_dir
    
    echo number of files: "${#files[@]}"
    echo $HOME
    echo $data_dir
    echo $output_dir
    echo $sub
    echo work directory is: $working_dir
    echo scratch directory is: $scratch_dir
    
    singularity run --bind $scratch_dir:/scratch --cleanenv $sing_image $data_dir $output_dir -w $working_dir participant --participant-label $sub --fs-no-reconall --fs-license-file $fs_lic --output-spaces MNI152NLin2009cAsym:res-2 --ignore slicetiming
    
    rm -rf $working_dir
    rm -rf $scratch_dir
fi

