# BC-ABCD-denoise
Denoising ABCD with deepcorr

#### Enviroment setup

Python & JupyterLab
```
python==3.8.17
jupyterlab==4.0.2
```

Python Packages:
```
torch==1.13.1+cu117
numpy==1.22.4
nibabel==3.2.1
matplotlib==3.3.2
scipy==1.7.3
sklearn==0.24.2
ants==0.3.8
seaborn==0.11.0
pandas==1.5.3
```

Easiest way to reproduce the enviroment is via conda

```
conda create -n deepcor python=3.8.17 jupyterlab=4.0.2
conda activate deepcor
pip install torch==1.13.1 numpy==1.22.4 nibabel==3.2.1 matplotlib==3.3.2 scipy==1.7.3 scikit-learn==0.24.2 antspyx seaborn==0.11.0 pandas==1.5.3
```

In case you don't have conda you can install it like this:

```
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
```

#### Instructions for Stefano to grab the forrest data

Once you clone the repository, in the repository root (BC-ABCD-denoise folder) run the following to first create the directories that will be needed: 
```
mkdir Data
mkdir ./Data/StudyForrest
mkdir ./Data/StudyForrest/fmriprep
mkdir ./Data/StudyForrest/DeepCor-outputs
mkdir ./Data/StudyForrest/pretraining
```

Then to copy the necessary StudyForrest files from my folder: 

```
cp -r ~/../aglinska/BC-ABCD-denoise/Data/StudyForrest/fmriprep/ ./Data/StudyForrest
cp -r ~/../aglinska/BC-ABCD-denoise/Data/StudyForrest/events/ ./Data/StudyForrest
cp -r ~/../aglinska/BC-ABCD-denoise/Data/StudyForrest/ROIs/ ./Data/StudyForrest
```


#### To set up a remote Jupyter session 

Request an interactive GPU job
```
srun --job-name=myGPUjob --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --time=04:00:00 --mem=64gb --partition=gpuv100 --pty bash -I
jupyter lab --no-browser --port=5678 --ip=gxxx # Replace gxxx with gpu node ID you get assigned (e.g. g003)
```

in a seperate terminal window, set up port forwarding: 
```
ssh -CNL 5678:gxxx:5678 <user>@andromeda.bc.edu # replace gxxx with node ID from previous step, and <user> with your username
```


