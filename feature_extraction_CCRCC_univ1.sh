#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1

# Run the script
python trident/run_batch_of_slides.py \
  --task all \
  --wsi_dir /mnt/pool/ovariancancer/raw_data/CCRCC \
  --job_dir /mnt/pool/ovariancancer/CCRCC_results/univ1 \
  --patch_encoder uni_v1 \
  --mag 20 \
  --patch_size 256 \
  --skip_errors

# nohup bash feature_extraction_CCRCC_univ1.sh > /home/jma/Documents/Beatrice/logs/feature_extraction_CCRCC_univ1_$(date +%Y%m%d_%H%M%S).log 2>&1 &