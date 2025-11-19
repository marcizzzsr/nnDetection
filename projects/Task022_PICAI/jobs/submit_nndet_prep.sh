#!/bin/bash
#SBATCH --job-name=nndet-prep
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Set environment variables for nnDetection
export det_data=/opt/data/scratch/datasets/nn-datasets
export det_num_threads=16

num_processes=6

# Run apptainer with bind mounts and nndet_train command
apptainer exec \
    --nv \
    --bind /mnt/data/unisr-data:/opt/data/scratch \
    --bind /home/agazzi.marcello@ihsr.dom/nnDetection:/opt/code/nndet \
    --env det_data=$det_data \
    --env det_num_threads=$det_num_threads \
    /home/agazzi.marcello@ihsr.dom/my_images/sif/nndetection.sif \
    python /opt/code/nndet/scripts/preprocess.py 022 \
        -o prep.plan=True \
        prep.crop=False \
        prep.analyze=False \
        prep.process=False

echo "Job finished at: $(date)"