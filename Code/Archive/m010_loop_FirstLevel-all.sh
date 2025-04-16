#!/bin/bash
#SBATCH --job-name=MATLAB-1stLevel
#SBATCH --output=../Data/SLURM-reports/MATLAB-1stLevel/MATLAB-1stLevel-outputs-_%a.txt
#SBATCH --error=../Data/SLURM-reports/MATLAB-1stLevel/MATLAB-1stLevel-errors-_%a.txt
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

subID=${subject_list[${SLURM_ARRAY_TASK_ID}]}
module load matlab


for rID in $(seq 1 2);
do

    # epi_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/${subID}/ses-baselineYear1Arm1/func/s8_${subID}_ses-baselineYear1Arm1_task-nback_run-0${rID}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii
    # analysis_dir=../analyses_SPM/s8_preproc_${rID}/${subID}
    
    # epi_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/${subID}/ses-baselineYear1Arm1/func/${subID}_ses-baselineYear1Arm1_task-nback_run-0${rID}_space-MNI152NLin2009cAsym_res-2_desc-COMPCORR_bold.nii
    # analysis_dir=../analyses_SPM/COMPCORR_${rID}/${subID}
    
    # epi_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/${subID}/ses-baselineYear1Arm1/func/${subID}_ses-baselineYear1Arm1_task-nback_run-0${rID}_space-MNI152NLin2009cAsym_res-2_desc-DEEPCOR_bold.nii
    # analysis_dir=../analyses_SPM/DEEPCOR_strict_${rID}/${subID}

    # epi_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/${subID}/ses-baselineYear1Arm1/func/${subID}_ses-baselineYear1Arm1_task-nback_run-0${rID}_space-MNI152NLin2009cAsym_res-2_desc-DEEPCOR_bold.nii
    # analysis_dir=../analyses_SPM/DEEPCOR_${rID}/${subID}



    # epi_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/${subID}/ses-baselineYear1Arm1/func/${subID}_ses-baselineYear1Arm1_task-nback_run-0${rID}_space-MNI152NLin2009cAsym_res-2_desc-DEEPCOR-TG_bold.nii
    # analysis_dir=../analyses_SPM/DEEPCOR_TG_${rID}/${subID}

    # epi_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/${subID}/ses-baselineYear1Arm1/func/${subID}_ses-baselineYear1Arm1_task-nback_run-0${rID}_space-MNI152NLin2009cAsym_res-2_desc-DEEPCOR-FG_bold.nii
    # analysis_dir=../analyses_SPM/DEEPCOR_FG_${rID}/${subID}
    
    # epi_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/${subID}/ses-baselineYear1Arm1/func/${subID}_ses-baselineYear1Arm1_task-nback_run-0${rID}_space-MNI152NLin2009cAsym_res-2_desc-DEEPCOR-twin_bold.nii
    # analysis_dir=../analyses_SPM/DEEPCOR_twin_${rID}/${subID}

    epi_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/020-fmriprepped/${subID}/ses-baselineYear1Arm1/func/${subID}_ses-baselineYear1Arm1_task-nback_run-0${rID}_space-MNI152NLin2009cAsym_res-2_desc-DEEPCOR-BG_bold.nii
    analysis_dir=../analyses_SPM/DEEPCOR_BG_${rID}/${subID}

    
    
    
    multicond_path=/mmfs1/data/aglinska/BC-ABCD-denoise/Data/030-subject-multiconds/${subID}_nback_multicond_run_${rID}.mat
    
    echo ${epi_path}
    echo ${analysis_dir}
    echo ${multicond_path}
    

    if [ -e $epi_path ]; then
        echo "EPI path exists"
    else 
        echo "!!!! EPI PATH DOES NOT EXIST !!!!"
        exit 1
    fi 
    
    if [ -e $multicond_path ]; then
        echo "multicond_path path exists"
    else 
        echo "!!!! multicond_path PATH DOES NOT EXIST !!!!"
        exit 1
    fi 

    mkdir -p $analysis_dir
    
    matlab -nodisplay -r "m010_func_fist_level_v2('${analysis_dir}','${epi_path}','${multicond_path}');exit"

done


