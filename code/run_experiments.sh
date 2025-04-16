#!/bin/bash
# This script runs all of the quantitative experiments for the paper in sequence on a single GPU and saves results to ~/wandb 
# Each experiment is run 5 times with different random seeds using 5-fold cross-validation. It will take a while to complete - parallelize for faster results.

set -euxo pipefail

# Set these as appropriate
ROOT_DIR=~/code/IV25-beyond-breathalysers
CUDA_DEVICE_TO_USE=0


# Main experiment
cd $ROOT_DIR
echo "Running main experiment (Fig 7)"
for seed in {1,2,3,4,5}; do 
    for exp in {gaze_tracking,silent_reading,choice_reaction,fixed_gaze}; do 
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_TO_USE python code/train.py --train_exp $exp --name lite$seed-$exp --seed $seed --cfg code/configs/lite.yaml; 
    done; 
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_TO_USE python code/train.py --name lite$seed-all --seed $seed --cfg code/configs/lite-all.yaml;
done


# CNN experiment
echo "Running CNN experiment"
for seed in {1,2,3,4,5}; do 
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_TO_USE python code/train.py --name lite$seed-all-cnn --seed $seed --cfg code/configs/lite-all-cnn.yaml;
done


# Varying sampling rate
echo "Running sampling rate experiment (Fig 8a)"
for seed in {1,2,3,4,5}; do 
    for sample_rate in {10,20,30}; do
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_TO_USE python code/train.py --name lite$seed-all-resample$sample_rate --seed $seed --cfg code/configs/lite-all-resample$sample_rate.yaml;
    done
done


# Varying window duration
echo "Running changing window experiment (Fig 8b)"
for seed in {1,2,3,4,5}; do 
    for window in {128,256,384}; do
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_TO_USE python code/train.py --name lite$seed-all-window$window --seed $seed --cfg code/configs/lite-all-window$window.yaml;
    done
done


# Varying input data dimension
echo "Running changing data dimension experiment (Fig 8c)"
for seed in {1,2,3,4,5}; do 
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_TO_USE python code/train.py --name lite$seed-all-event-only --seed $seed --cfg code/configs/lite-all-event-only.yaml;
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_TO_USE python code/train.py --name lite$seed-all-pd --seed $seed --cfg code/configs/lite-all-pd.yaml;
done
