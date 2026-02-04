#!/bin/bash

# conda activate ptcu117

INVIDS=(/media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/All_videos/2022_12_16_09_21_39_Movie.mp4)

cd ..

for INVID in ${INVIDS[@]};
do
    echo ${INVID}
    python scripts/pose_estimation_topdown.py \
        demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
        https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
        configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py \
        https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth \
        --input ${INVID} \
        --output-root=vis_results/panoptic --show --draw-heatmap --save-predictions

done

# configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py \
# https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth \