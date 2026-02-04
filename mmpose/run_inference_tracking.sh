#!/bin/bash

source activate ptcu117

INVIDS=(/media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/All_videos/2022_12_16_09_21_39_Movie.mp4)

for INVID in ${INVIDS[@]};
do
    echo ${INVID}
    python demo/topdown_demo_with_mmdet.py \
        demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
        https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
        configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
        https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
        --video-path ${INVID} \
        --smooth --smooth-filter-cfg \
        --out-video-root=vis_results/tracking_demo --show --draw-heatmap

done
