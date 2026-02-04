#!/bin/bash

# --- 1. Argument Validation (Best Practice) ---
# Exit if the required number of arguments is not provided.
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <root_dir> <save_dir> <save_name> <gpu_device_id>"
    exit 1
fi

root_dir=$1
save_dir=$2
save_name=$3
device=$4

# --- 2. Use Absolute Paths & Define Variables for Clarity (Best Practice) ---
# Using variables for long paths makes the command block much cleaner.
CONDA_PYTHON="~/miniconda3/envs/anonymization/bin/python3"
PROJECT_ROOT="OR_anonymization"
MODEL_WEIGHT="weights/4dor_iter1.pth"

# Define expected file paths based on the pipeline logic
DETECTION_NAME="${save_name}.lmdb"
DETECTION_PATH="${save_dir}/detections/${save_name}.lmdb"
TRACKING_PATH="${save_dir}/tracking/${save_name}_tracking.lmdb"
# POTENTIAL FIX: Corrected MVA output filename assumption
MVA_PATH="${save_dir}/mva/${save_name}_mva_tracking_reid.lmdb"
MVA_ID_PATH="${save_dir}/mva/${save_name}_mva_tracking_reid_id.lmdb"

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
    echo "--- Starting full-body detection ---" && \
    cd '${PROJECT_ROOT}/detector' && \
    \
    '${CONDA_PYTHON}' inference_lmdb_4dor.py --num_queries 1000 --epochs 50 --enc_layers 6 --dec_layers 6 \
       --with_box_refine  --lr_drop 40 --batch_size 1 --aps 1       \
       --output_dir aps_swinl --backbone swin-l --use_checkpoint \
       --root_dir '${root_dir}' \
       --save_dir '${save_dir}' \
       --save_name '${DETECTION_NAME}' \
       --weight '${MODEL_WEIGHT}' && \
    \
    echo "--- Starting tracking ---" && \
    cd '${PROJECT_ROOT}/mva' && \
    \
    '${CONDA_PYTHON}' ssl/tracking_lmdb_4dor.py --path '${DETECTION_PATH}' \
       --save_dir '${save_dir}' \
       --root_dir '${root_dir}' && \
    \
    echo "--- Starting multi-view association ---" && \
    '${CONDA_PYTHON}' ssl/mva_tracking_lmdb_4dor.py --cfg configs/4dor_iter1.yaml \
       --path '${DETECTION_PATH}' \
       --tracking_path '${TRACKING_PATH}' \
       --save_dir '${save_dir}' \
       --save_suffix _mva_tracking_reid.lmdb \
       --root_dir '${root_dir}' && \
    \
    echo "--- Starting assigning tracking id ---" && \
    '${CONDA_PYTHON}' ssl/assign_tracking_id_4dor.py --path '${MVA_PATH}' \
       --save_dir '${save_dir}' \
       --root_dir '${root_dir}' && \
    \
    echo "--- Starting pose estimation ---" && \
    cd '${PROJECT_ROOT}/mmpose' && \
    \
    '${CONDA_PYTHON}' pipeline_lmdb_4dor.py \
    configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py \
    rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth \
    --box_thresh 0.0 \
    --path '${MVA_ID_PATH}' \
    --root_dir '${root_dir}' \
    --save_suffix .lmdb && \
    \
    echo "--- All tasks completed successfully. ---" \
' > ${LOG_FILE} 2>&1 &

echo "Script has been launched in the background. You can now close this terminal."
