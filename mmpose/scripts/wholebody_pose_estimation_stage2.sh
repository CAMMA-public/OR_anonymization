cd ..
nohup python scripts/wholebody_pose_estimation_stage2.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth \
    --input /media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/All_videos/2022_12_16_09_21_39_Movie.mp4 \
    --output-root=vis_results/panoptic --show --draw-heatmap --save-predictions > output.log 2>&1 &