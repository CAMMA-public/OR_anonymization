import os
import pickle

for i in range(1, 11):
    img_dir = f'4D-OR/export_holistic_take{i}_processed/colorimage'
    img_names = os.listdir(img_dir)
    img_names.sort()

    img_dicts = []
    for c in range(6):
        count = 0
        img_dict = {}
        for img_name in img_names:
            if img_name[7] != str(c+1):
                continue
            img_dict[count] = img_name
            count += 1
        img_dicts.append(img_dict)

    with open(os.path.join(f'4D-OR/export_holistic_take{i}_processed/img_dicts.pkl'), 'wb') as f:
        pickle.dump(img_dicts, f)