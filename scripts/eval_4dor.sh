#!/bin/bash

# Usage:
# ./run_eval.sh <save_name> \
#     --pred_path <path1> [path2 ...] \
#     --anno_path <path1> [path2 ...] \
#     --img_dicts <path1> [path2 ...]

if [ "$#" -lt 7 ]; then
    echo "Usage:"
    echo "$0 <save_name> \\"
    echo "    --pred_path <path1> [path2 ...] \\"
    echo "    --anno_path <path1> [path2 ...] \\"
    echo "    --img_dicts <path1> [path2 ...]"
    exit 1
fi

save_name=$1
shift

pred_paths=()
anno_paths=()
img_dicts=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pred_path)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                pred_paths+=("$1")
                shift
            done
            ;;
        --anno_path)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                anno_paths+=("$1")
                shift
            done
            ;;
        --img_dicts)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                img_dicts+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check inputs
if [ ${#pred_paths[@]} -eq 0 ] || \
   [ ${#anno_paths[@]} -eq 0 ] || \
   [ ${#img_dicts[@]} -eq 0 ]; then
    echo "Error: --pred_path, --anno_path, and --img_dicts are required."
    exit 1
fi

if [ ${#pred_paths[@]} -ne ${#anno_paths[@]} ] || \
   [ ${#pred_paths[@]} -ne ${#img_dicts[@]} ]; then
    echo "Error: Number of pred_path, anno_path, and img_dicts must match."
    exit 1
fi

PYTHON="$HOME/miniconda3/envs/anonymization/bin/python3"

LOG_FILE="${save_name}_evaluation.log"

echo "Starting evaluation..."
echo "Number of datasets: ${#pred_paths[@]}"
echo "Log: ${LOG_FILE}"
echo "Monitor with: tail -f ${LOG_FILE}"

(
    cd "evaluation" || exit 1

    "$PYTHON" -u robust_evaluation_4dor.py \
        --pred_path "${pred_paths[@]}" \
        --anno_path "${anno_paths[@]}" \
        --img_dicts "${img_dicts[@]}" \
        --kpt_thresh 1.0 \
        --vis_thresh 0.4 \
        --out_of_body
) > "$LOG_FILE" 2>&1 &

echo "Evaluation launched in background."