#!/bin/bash
#SBATCH --job-name=nndet-train
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
export det_num_threads=8
export MLFLOW_TRACKING_URI=/runs

# https://github.com/MIC-DKFZ/nnDetection/issues/151
# max_num_epochs=250
# num_train_batches_per_epoch=500
# batch_size=16

# Run apptainer with bind mounts and nndet_train command
apptainer exec \
    --nv \
    --bind /mnt/scratch/:/opt/data/scratch \
    --bind /home/agazzi.marcello@ihsr.dom/nnDetection/nndet/ptmodule/:/opt/code/nndet/nndet/ptmodule \
    --bind $HOME/runs:/runs \
    --env det_data=$det_data \
    --env det_models=$det_models \
    --env det_num_threads=$det_num_threads \
    --env MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
    /home/agazzi.marcello@ihsr.dom/my_images/sif/nndetection.sif \
    nndet_train 022
        # -o trainer_cfg.max_num_epochs=$max_num_epochs \
        # trainer_cfg.num_train_batches_per_epoch=$num_train_batches_per_epoch \
        # +augment_cfg.batch_size=$batch_size 


pkill -P $$ 
echo "Job finished at: $(date)"