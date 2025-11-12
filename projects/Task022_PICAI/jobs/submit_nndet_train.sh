#!/bin/bash
#SBATCH --job-name=nndet-train-scratch
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=nndet-train-scratch_%j.log
#SBATCH --error=nndet-train-scratch_%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=agazzi.marcello@hsr.it

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Set environment variables for nnDetection
export det_data=/opt/data/scratch/unisr-data
export det_models=/opt/data/scratch/unisr-data/models
export det_num_threads=12

# Run apptainer with bind mounts and nndet_train command
apptainer exec \
    --nv \
    --bind /mnt/scratch/:/opt/data/scratch \
    --env det_data=$det_data \
    --env det_models=$det_models \
    /home/agazzi.marcello@ihsr.dom/my_images/sif/nndetection.sif \
    nndet_train 022

echo "Job finished at: $(date)"