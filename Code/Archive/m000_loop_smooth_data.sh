#!/bin/bash
#SBATCH --job-name=MATLAB-smooth
#SBATCH --output=../Data/SLURM-reports/MATLAB-smooth/MATLAB-smooth-outputs-_%a.txt
#SBATCH --error=../Data/SLURM-reports/MATLAB-smooth/MATLAB-smooth-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=64gb
#SBATCH --array=0-34


#python py_get_fmriprepped_subs_list.py # Write subjects to subject_list.txt file 

subject_list=()
while IFS= read -r line; do # # Read the file line by line and add each line (filename) to the array
    subject_list+=("$line")
done < subject_list.txt

echo Found ${#subject_list[@]} subjects
echo subNum is ${SLURM_ARRAY_TASK_ID}
echo subID is ${subject_list[${SLURM_ARRAY_TASK_ID}]}

module load matlab

echo "m000_spm_smooth_data('${subject_list[${SLURM_ARRAY_TASK_ID}]}',1);exit"
matlab -nodisplay -r "m000_spm_smooth_data('${subject_list[${SLURM_ARRAY_TASK_ID}]}',1);exit"

echo "m000_spm_smooth_data('${subject_list[${SLURM_ARRAY_TASK_ID}]}',2);exit"
matlab -nodisplay -r "m000_spm_smooth_data('${subject_list[${SLURM_ARRAY_TASK_ID}]}',2);exit"

