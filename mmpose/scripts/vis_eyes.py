import os
import cv2
import pickle

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

vid_dir = '/media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/cut_CH3'
pose_dir = '/media/chen/hdd1/chen/datasets/BERN_OR_VIDEOS/ByteTrack/wholebody_pose'
pose_files = os.listdir(pose_dir)
for pose_file in pose_files:
    pose_name, suffix = pose_file.rsplit('.', 1)
    print(pose_name)
    if suffix != 'pkl':
        continue
    pose_path = os.path.join(pose_dir, pose_file)
    with open(pose_path, 'rb') as handle:
        pose_dict = pickle.load(handle)
    
    vid_path = os.path.join(vid_dir, pose_name + '.mp4')
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while 1:
        ret, frame = cap.read()
        if ret:
            if count not in pose_dict:
                count += 1
                continue
            lines = pose_dict[count]
            for i, line in enumerate(lines):
                # len(line): 58
                x1 = int(line[2])
                y1 = int(line[3])
                x2 = int(line[4])
                y2 = int(line[5])
                kpts = line[7:273]
                kpts_score = line[273:]
                color = get_color(abs(i))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=3)
                for j in range(59, 71):
                    kpt_x = int(kpts[j*2])
                    kpt_y = int(kpts[j*2+1])
                    cv2.circle(frame, (kpt_x, kpt_y), radius=3, color=color, thickness=-1)
                left_kpt_s_sum = 0
                for j in range(59, 65):
                    kpt_s = kpts_score[j]
                    left_kpt_s_sum += kpt_s
                right_kpt_s_sum = 0
                for j in range(65, 71):
                    kpt_s = kpts_score[j]
                    right_kpt_s_sum += kpt_s
                left_kpt_s = left_kpt_s_sum / 6
                right_kpt_s = right_kpt_s_sum / 6
                cv2.putText(frame, str(round(left_kpt_s, 2)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, str(round(right_kpt_s, 2)), (int(x2), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join('vis_results', 'eyes', str(count) + '.jpg'), frame)
            count += 1
            if count > 100:
                exit(1)
        else:
            break