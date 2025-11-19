#!/bin/bash
#SBATCH --job-name=nndet-unpack
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Set environment variables for nnDetection
export det_data=/opt/data/scratch/unisr-data/datasets/nn-datasets
export det_models=/opt/data/scratch/unisr-data/models
export det_num_threads=16
export MLFLOW_TRACKING_URI=/opt/data/scratch/unisr-data/exp-runs

num_processes=6


# Run apptainer with bind mounts and nndet_train command
apptainer exec \
    --bind /mnt/beegfs/data/:/opt/data/scratch \
    --bind /home/agazzi.marcello@ihsr.dom/nnDetection:/opt/code/nndet \
    --env det_data=$det_data \
    --env det_models=$det_models \
    --env det_num_threads=$det_num_threads \
    --env MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
    /home/agazzi.marcello@ihsr.dom/my_images/sif/nndetection.sif \
    nndet_unpack $det_data/Task022_PICAI/preprocessed/D3V001_3d/imagesTr $num_processes
echo "Job finished at: $(date)"