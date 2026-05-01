import io
import os
from zipfile import ZipFile

from PIL import Image
import colmap_io
from tqdm import tqdm
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import gzip

Image.MAX_IMAGE_PIXELS = 933120000

def compute_cosine_of_angle(R1, R2):
    """Compute the cosine of the angle between two rotation matrices."""
    # Compute the relative rotation matrix
    R_rel = R2 @ R1.T  # Relative rotation

    # Compute the cosine of the angle using the trace of the relative rotation matrix
    cos_angle = (np.trace(R_rel) - 1) / 2
    
    return cos_angle

# Computes IoU (like in CroCoV2 https://arxiv.org/pdf/2211.10408)
def find_best_next_image(current_id, visible_3D_ids, image_to_3Dpoints, images_file, used_images):
    """Find the best next image to continue the sequence"""
    best_next_id = None
    best_score = float('-inf')

    for other_id, other_3D_ids in image_to_3Dpoints.items():
        if other_id == current_id or other_id in used_images:
            continue  # Skip self-comparison and already used images

        shared_points = visible_3D_ids.intersection(other_3D_ids)
        iou = len(shared_points) / len(visible_3D_ids) if len(visible_3D_ids) > 0 else 0

        R1 = R.from_quat(images_file[current_id].qvec, scalar_first=True).as_matrix()
        R2 = R.from_quat(images_file[other_id].qvec, scalar_first=True).as_matrix()
        cos = compute_cosine_of_angle(R1, R2)
        angle_diff = np.degrees(np.arccos(np.clip(cos, -1, 1)))

        if angle_diff < 10:
            continue

        score = iou * 4 * cos * (1 - cos)
        
        if iou >= 0.10:  # At least 10% overlap
            if score > best_score:
                best_score = score
                best_next_id = other_id

    return best_next_id

def build_sequence_from_seed(seed_id, image_to_3Dpoints, images_file, used_images, min_sequence_length=3):
    """Build a sequence starting from a seed image"""
    sequence = [seed_id]
    used_images.add(seed_id)
    current_id = seed_id
    
    while True:
        current_visible_3D_ids = image_to_3Dpoints[current_id]
        next_id = find_best_next_image(current_id, current_visible_3D_ids, image_to_3Dpoints, images_file, used_images)
        
        if next_id is None:
            break
            
        sequence.append(next_id)
        used_images.add(next_id)
        current_id = next_id
    
    # Only return sequences that meet minimum length requirement
    if len(sequence) >= min_sequence_length:
        return sequence
    else:
        # If sequence is too short, remove images from used_images so they can be used elsewhere
        for img_id in sequence:
            used_images.discard(img_id)
        return None

def find_sequences(image_to_3Dpoints, images_file, min_sequence_length=3):
    """Find all sequences in the image set"""
    used_images = set()
    sequences = []
    
    # Sort images by number of visible 3D points (descending) to start with well-connected images
    sorted_images = sorted(
        image_to_3Dpoints.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    for image_id, _ in sorted_images:
        if image_id in used_images:
            continue
            
        sequence = build_sequence_from_seed(image_id, image_to_3Dpoints, images_file, used_images, min_sequence_length)
        if sequence is not None:
            sequences.append(sequence)
    
    return sequences

def get_2d_points_for_3d_ids(image_data, point3d_ids_to_find):
    id_to_idx = {id: idx for idx, id in enumerate(image_data.point3D_ids)}
    indices = np.array([id_to_idx.get(id, -1) for id in point3d_ids_to_find])
    valid_mask = indices != -1
    result = np.full((len(point3d_ids_to_find), 2), np.nan)
    result[valid_mask] = image_data.xys[indices[valid_mask]]
    return result

if __name__ == "__main__":
    dataroot = '/mimer/NOBACKUP/Datasets/MegaScenes/v1.0'
    zfpath_reconstruct = os.path.join(dataroot, "reconstruct.zip")
    zfpath_images = os.path.join(dataroot, "images.zip")

    print('Opening up the zip files...')

    out = {}

    with ZipFile(zfpath_reconstruct) as zf:
        file_list = zf.namelist()
        imgs_list: list[str] = [
                path for path in zf.namelist()
                if path.endswith("images.bin")
        ]
        
        zf_img = ZipFile(zfpath_images)
        for images_file_path in tqdm(imgs_list):
            split_path = images_file_path.split('/')
            scene_id_1, scene_id_2, reconstruction_id = split_path[1], split_path[2], split_path[4]

            def get_im_path(name):
                return str(Path('images') / scene_id_1 / scene_id_2 / name)

            images_file = colmap_io.read_images_binary_input(io.BytesIO(zf.read(images_file_path)))

            if len(images_file.keys()) < 12:
                # Optional
                # If it is a very small scene, skip it... 
                continue

            


            sequence = []
            for img_id, img_data in images_file.items():
                sequence.append(
                    {
                        'filepath': get_im_path(img_data.name),
                        'id': img_id,
                        'qvec': img_data.qvec.tolist(),
                        'tvec': img_data.tvec.tolist(),
                    }
                )

            out[f"{scene_id_1}_{scene_id_2}_{reconstruction_id}"] = sequence
            print(f"Processed scene {scene_id_1}/{scene_id_2}/{reconstruction_id} with {len(sequence)} images")

        with open("output.json", "w") as f:
            json.dump(out, f, indent=4)  # `indent=4` makes it pretty-printed

        # Save as .jgz
        with gzip.open("output.jgz", "wt", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=4)
        
        zf_img.close()