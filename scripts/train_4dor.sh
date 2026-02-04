#!/bin/bash

# --- 1. Argument Validation (Best Practice) ---
# Exit if the required number of arguments is not provided.
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <root_dir> <save_dir> <save_name> <gpu_device_id>"
    exit 1
fi

root_dir=$1
pred_path=$2
save_root_dir=$3
device=$4

# --- 2. Use Absolute Paths & Define Variables for Clarity (Best Practice) ---
# Using variables for long paths makes the command block much cleaner.
CONDA_PYTHON="~/miniconda3/envs/anonymization/bin/python3"
PROJECT_ROOT="OR_anonymization"

export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_VISIBLE_DEVICES=$device

# --- 3. Create a Unique Log File Name (Robustness) ---
LOG_FILE="${save_name}_pipeline.log"

echo "Starting background inference pipeline..."
echo "GPU Device: ${device}"
echo "Output from both tasks will be saved to ${LOG_FILE}"
echo "You can monitor the progress with: tail -f ${LOG_FILE}"

# --- The Corrected Sequential Command ---
# We use bash -c to group the commands together for nohup.
# The '&&' ensures the second command only runs if the first succeeds.
nohup bash -c ' \
    echo "--- Generate pseudo labels for fine-tuning detector ---" && \
    cd '${PROJECT_ROOT}/detector' && \
    \
    '${CONDA_PYTHON}' generate_pseudo_labels_4dor.py \
       --pred_path '${pred_path}' \
       --root_dir '${root_dir}' \
       --save_root_dir '${save_root_dir}' && \
    \
    echo "--- Generate coco annotations ---" && \
    \
    '${CONDA_PYTHON}' merge_coco_files.py && \
    \
    echo "--- Fine-tuning full-body detector ---" && \
    \
    '${CONDA_PYTHON}' main.py \
    --num_queries 1000 --epochs 20 --enc_layers 6 --dec_layers 6 \
    --with_box_refine  --lr_drop 10 --batch_size 1 --aps 1       \
    --output_dir aps_swinl_4dor --backbone swin-l --use_checkpoint && \
    \
    cd '${PROJECT_ROOT}/mva' && \
    \
    echo "--- Training multi-view association model ---" && \
    '${CONDA_PYTHON}' ssl/main.py --cfg configs/4dor_iter1.yaml && \
    \
    echo "--- All tasks completed successfully. ---" \
' > ${LOG_FILE} 2>&1 &

echo "Script has been launched in the background. You can now close this terminal."
