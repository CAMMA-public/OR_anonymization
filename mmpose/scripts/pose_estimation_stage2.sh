INVID=/media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/All_videos/2022_12_16_09_21_39_Movie.mp4
cd ..
nohup python scripts/pose_estimation_stage2.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth \
    --input ${INVID} \
    --output-root=vis_results/panoptic --show --draw-heatmap --save-predictions > output.log 2>&1 &