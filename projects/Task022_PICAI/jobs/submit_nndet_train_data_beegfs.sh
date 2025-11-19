#!/bin/bash
#SBATCH --job-name=nndet-train-data-beegfs
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=agazzi.marcello@hsr.it

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Set environment variables for nnDetection
export det_data=/opt/data/scratch/unisr-data/datasets/nn-datasets
export det_models=/opt/data/scratch/unisr-data/models
export det_num_threads=16
export MLFLOW_TRACKING_URI=/opt/data/scratch/unisr-data/exp-runs/

# https://github.com/MIC-DKFZ/nnDetection/issues/151
max_num_epochs=1
num_train_batches_per_epoch=500
batch_size=8

# Run apptainer with bind mounts and nndet_train command
apptainer exec \
    --nv \
    --bind /mnt/beegfs/data/:/opt/data/scratch \
    --bind /home/agazzi.marcello@ihsr.dom/nnDetection:/opt/code/nndet \
    --env det_data=$det_data \
    --env det_models=$det_models \
    --env det_num_threads=$det_num_threads \
    --env MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
    /home/agazzi.marcello@ihsr.dom/my_images/sif/nndetection.sif \
    nndet_train 022 \
        -o exp.tag="beegfs-scratch-bench" \
        trainer_cfg.max_num_epochs=$max_num_epochs \
        trainer_cfg.num_train_batches_per_epoch=$num_train_batches_per_epoch \
        trainer_cfg.swa_epochs=0 \
        +augment_cfg.batch_size=$batch_size
        
echo "Job finished at: $(date)"