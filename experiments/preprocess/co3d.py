# Go through CO3D and simply save all the files unzipped

import gzip
import json
import os.path as osp
import os
import logging
import io
from zipfile import ZipFile
from tqdm import tqdm
from PIL import Image
def resize_and_crop_center(img, target_size=480):
    # Scale so shortest side = target_size
    scale = target_size / min(img.size)
    new_size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(new_size, Image.LANCZOS)

    # Center crop to (target_size, target_size)
    left = (img.width - target_size) // 2
    top = (img.height - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    return img.crop((left, top, right, bottom))

SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]
category = sorted(SEEN_CATEGORIES)
min_num_images = 24
OUTPUT_DIR = "/mimer/NOBACKUP/groups/3d-dl/co3d"
DATA_DIR="/mimer/NOBACKUP/Datasets/CO3D"
ANNOTATION_DIR = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/co3d_anno/cleaned"
TARGET_SIZE = 480
data_store = {}
os.makedirs(OUTPUT_DIR, exist_ok=True)
for c in category:
    for split_name in ["train"]:
        annotation_file = osp.join(
            ANNOTATION_DIR, f"{c}_{split_name}.jgz"
        )

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {annotation_file}")
            continue

        for seq_name, seq_data in annotation.items():
            if len(seq_data) < min_num_images:
                continue

            data_store[seq_name] = seq_data


def _collect_zip_paths():
    zip_paths = {}
    for filename in os.listdir(DATA_DIR):
        if filename.startswith("CO3D_") and filename.endswith(".zip"):
            object_class = filename[5:-4]
            zip_paths[object_class] = osp.join(DATA_DIR, filename)
    return zip_paths


def _get_zipfile(object_class):
    if object_class not in _zipfiles:
        _zipfiles[object_class] = ZipFile(zip_paths[object_class], 'r')
    return _zipfiles[object_class]


def _read_image_from_zip(filepath):
    object_class = filepath.split('/')[0]

    if object_class not in zip_paths:
        raise ValueError(f"No zip path found for object class: {object_class}")

    zf = _get_zipfile(object_class)
    img_data = zf.read(filepath)
    return Image.open(io.BytesIO(img_data)).convert("RGB")

zip_paths = _collect_zip_paths()
_zipfiles = {}  
for seq_name, seq_data in tqdm(data_store.items(), desc="Processing sequences"):
    for anno in seq_data:
        filepath = anno["filepath"]
        image_path = osp.join(DATA_DIR, filepath)
        image = _read_image_from_zip(filepath)

        # image_resized = image.resize(TARGET_SIZE, Image.LANCZOS)
        image_resized = resize_and_crop_center(image, TARGET_SIZE)
        save_path = osp.join(OUTPUT_DIR, filepath)
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        image_resized.save(save_path, format="JPEG", quality=95)

