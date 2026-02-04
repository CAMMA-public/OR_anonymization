import json
import os
import datetime
from collections import defaultdict

def merge_coco_files(coco_file_paths, output_path):
    """
    Merges multiple COCO-format JSON annotation files into a single file.

    Args:
        coco_file_paths (list): A list of paths to the COCO JSON files to merge.
        output_path (str): Path to save the merged COCO JSON file.
    """
    if not coco_file_paths:
        print("No COCO files provided to merge.")
        return

    # Initialize the structure for the merged COCO file
    merged_coco = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # --- Step 1: Load the first file to get initial info and licenses ---
    try:
        with open(coco_file_paths[0], 'r') as f:
            first_coco_data = json.load(f)
        merged_coco["info"] = first_coco_data.get("info", {
            "description": "Merged COCO Dataset",
            "year": datetime.date.today().year,
            "contributor": "Merger Script",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        })
        # For licenses, we can try to merge them or just take from the first file.
        # For simplicity, taking from the first. A more robust merge would de-duplicate.
        merged_coco["licenses"] = first_coco_data.get("licenses", [])
    except Exception as e:
        print(f"Error loading base info/licenses from {coco_file_paths[0]}: {e}")
        # Provide default info if first file fails or lacks info
        merged_coco["info"] = {
            "description": "Merged COCO Dataset",
            "year": datetime.date.today().year,
            "contributor": "Merger Script",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }
        merged_coco["licenses"] = [{"id": 1, "name": "Default License", "url": ""}]


    # --- Step 2: Merge Categories ---
    # Categories are merged based on their 'name'.
    # We create a new set of category IDs for the merged file.
    merged_category_name_to_id = {}
    next_merged_category_id = 1

    # This will map (original_file_index, original_category_id) -> new_merged_category_id
    original_cat_id_to_new_cat_id_map = {}

    for file_idx, coco_file_path in enumerate(coco_file_paths):
        try:
            with open(coco_file_path, 'r') as f:
                data = json.load(f)
            for category in data.get("categories", []):
                cat_name = category["name"]
                original_cat_id = category["id"]

                if cat_name not in merged_category_name_to_id:
                    merged_category_name_to_id[cat_name] = next_merged_category_id
                    merged_coco["categories"].append({
                        "id": next_merged_category_id,
                        "name": cat_name,
                        "supercategory": category.get("supercategory", "object") # Preserve supercategory
                    })
                    current_new_cat_id = next_merged_category_id
                    next_merged_category_id += 1
                else:
                    current_new_cat_id = merged_category_name_to_id[cat_name]

                original_cat_id_to_new_cat_id_map[(file_idx, original_cat_id)] = current_new_cat_id
        except Exception as e:
            print(f"Error processing categories from {coco_file_path}: {e}")
            continue

    # --- Step 3: Merge Images and Annotations ---
    # Images are merged based on 'file_name'.
    # New image IDs and annotation IDs are generated.
    merged_filename_to_image_id = {}
    next_merged_image_id = 1
    next_merged_annotation_id = 1

    for file_idx, coco_file_path in enumerate(coco_file_paths):
        try:
            with open(coco_file_path, 'r') as f:
                data = json.load(f)

            # Map original image IDs in this file to new merged image IDs
            current_file_original_img_id_to_new_img_id = {}

            for image_info in data.get("images", []):
                original_image_id = image_info["id"]
                file_name = image_info["file_name"]

                if file_name not in merged_filename_to_image_id:
                    # This is a new image (by file_name)
                    new_img_id = next_merged_image_id
                    merged_filename_to_image_id[file_name] = new_img_id

                    # Copy image info, update ID
                    new_image_info = image_info.copy()
                    new_image_info["id"] = new_img_id
                    # Ensure license ID is valid if present, or remove/default it
                    if "license" in new_image_info and new_image_info["license"] not in [lic["id"] for lic in merged_coco["licenses"]]:
                        # Simplistic: assign first license or remove
                        if merged_coco["licenses"]:
                            new_image_info["license"] = merged_coco["licenses"][0]["id"]
                        else:
                            del new_image_info["license"]

                    merged_coco["images"].append(new_image_info)
                    next_merged_image_id += 1
                else:
                    # Image (by file_name) already exists, use its existing new ID
                    new_img_id = merged_filename_to_image_id[file_name]

                current_file_original_img_id_to_new_img_id[original_image_id] = new_img_id

            # Now process annotations for this file
            for ann_info in data.get("annotations", []):
                original_img_id = ann_info["image_id"]
                original_cat_id = ann_info["category_id"]

                # Get the new image_id for this annotation
                if original_img_id not in current_file_original_img_id_to_new_img_id:
                    print(f"Warning: Annotation in {coco_file_path} references image_id {original_img_id} "
                          f"which was not found in its 'images' list. Skipping annotation.")
                    continue
                new_ann_image_id = current_file_original_img_id_to_new_img_id[original_img_id]

                # Get the new category_id for this annotation
                map_key = (file_idx, original_cat_id)
                if map_key not in original_cat_id_to_new_cat_id_map:
                    # This can happen if a category was defined in categories but an annotation
                    # uses a category_id not in that file's categories list, or if the category
                    # itself was corrupt and skipped.
                    # Or, if the category_id was valid but an error occurred during category processing for this file.
                    cat_name_for_error = "unknown"
                    for cat_entry in data.get("categories", []):
                        if cat_entry["id"] == original_cat_id:
                            cat_name_for_error = cat_entry["name"]
                            break
                    print(f"Warning: Annotation in {coco_file_path} (image_id {original_img_id}) "
                          f"references category_id {original_cat_id} (name: {cat_name_for_error}) "
                          f"which couldn't be mapped to a merged category. Skipping annotation.")
                    continue
                new_ann_category_id = original_cat_id_to_new_cat_id_map[map_key]


                # Copy annotation info, update IDs
                new_ann_info = ann_info.copy()
                new_ann_info["id"] = next_merged_annotation_id
                new_ann_info["image_id"] = new_ann_image_id
                new_ann_info["category_id"] = new_ann_category_id

                merged_coco["annotations"].append(new_ann_info)
                next_merged_annotation_id += 1

        except Exception as e:
            print(f"Error processing images/annotations from {coco_file_path}: {e}")
            continue

    # --- Step 4: Save the merged COCO file ---
    try:
        with open(output_path, 'w') as f:
            json.dump(merged_coco, f, indent=4)
        print(f"\nSuccessfully merged {len(coco_file_paths)} COCO files into: {output_path}")
        print(f"Merged dataset contains:")
        print(f"  - {len(merged_coco['images'])} images")
        print(f"  - {len(merged_coco['annotations'])} annotations")
        print(f"  - {len(merged_coco['categories'])} categories")
    except Exception as e:
        print(f"Error saving merged COCO file to {output_path}: {e}")

# --- Example Usage ---
if __name__ == '__main__':
    # Create some dummy COCO JSON files for testing

    file1_path = "/media/camma-monitor/Storage_postprocessing2/pose_detection/CrowdHuman/annotations/train.json"
    file2_path = "006.json"
    file3_path = "007.json"
    file4_path = "009.json"
    merged_output_path = "/media/camma-monitor/Storage_postprocessing2/pose_detection/CrowdHuman/annotations/train_monitor10k_only_score_06.json"

    files_to_merge = [file1_path, file2_path, file3_path, file4_path]
    merge_coco_files(files_to_merge, merged_output_path)

    # You can then inspect "temp_coco_merge_test/merged_coco.json"
    # Expected:
    # - Info from coco1.json (or default)
    # - Licenses from coco1.json (or default)
    # - Categories: "cat", "dog", "person" with new unique IDs (e.g., 1, 2, 3)
    # - Images: "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg" with new unique IDs.
    #   "img2.jpg" should only appear once.
    # - Annotations: All 5 original annotations, but with updated image_ids and category_ids,
    #   and new unique annotation_ids.
