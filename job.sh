#!/bin/sh -l
# FILENAME: my_job

#SBATCH -A long
#SBATCH --nodes=1 --gpus-per-node=1 --mem=49GB
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/notebooks/outputs.txt
#SBATCH --error=/scratch/gilbreth/akamsali/Research/Makin/Auditory_Cortex/notebooks/error.txt
#SBATCH --job-name layer_0

# Print the hostname of the compute node on which this job is running.
/bin/hostname

. ~/.bashrc
conda activate research_env

python /scratch/gilbreth/akamsali/Research/Makin/ECE695-finalproject/main.py
