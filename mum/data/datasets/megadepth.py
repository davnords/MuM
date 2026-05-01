import gzip
import json
import os.path as osp
import logging

import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset
from PIL import UnidentifiedImageError

class MegaDepthDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_DIR: str = None,
        ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        sampling_weight: int = 3,
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        
        self.duplicate_img = False
        self.load_depth = False

        if DATA_DIR is None or ANNOTATION_DIR is None:
            raise ValueError("Both DATA_DIR and ANNOTATION_DIR must be specified.")

        
        self.sampling_weight = sampling_weight
        if split == "train":
            split_name = "train.jgz"
        elif split == "test":
            split_name = "val.jgz"
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_scenes = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"MegaDepth directory is {DATA_DIR}")

        self.DATA_DIR = DATA_DIR
        self.ANNOTATION_DIR = ANNOTATION_DIR

        total_frame_num = 0

        annotation_file = osp.join(
            self.ANNOTATION_DIR, "megadepth", split_name
        )

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {annotation_file}")

        for scene_name, scene_data in annotation.items():
            if scene_name in self.invalid_scenes:
                    continue
            for i, seq_data in enumerate(scene_data):
                if len(seq_data) < min_num_images:
                    continue
                
                total_frame_num += len(seq_data)
                self.data_store[f"{scene_name}_{i}"] = seq_data

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        
        # This is an unfair number of total frames because we have repition of the same frames in the sequences.
        # self.total_frame_num = total_frame_num
        # self.total_frame_num = self.sequence_list_len # I believe this is around 115k 
        self.total_frame_num = 204684 # This is the number of unique images in the dataset, I have counted the unique annotations

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: MegaDepth Data size: {self.sequence_list_len}")
        logging.info(f"{status}: MegaDepth Data length of training set: {len(self)}")
        logging.info(f"{status}: MegaDepth Frame number: {total_frame_num}")

        del annotation

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        
        # print('Aspect ratio: ', aspect_ratio)

        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(
                len(metadata), img_per_seq, replace=self.duplicate_img
            )

        annos = [metadata[i] for i in ids]

        images = []
        image_paths = []

        for anno in annos:
            while True:
                filepath = anno["filepath"]
                image_path = filepath

                try:
                    image = Image.open(self.DATA_DIR+'/'+image_path).convert("RGB")
                    images.append(image)
                    image_paths.append(image_path)
                    break
                except UnidentifiedImageError:
                    logging.error(f"[WARN] Bad image {image_path}, resampling...")
                    new_idx = random.randint(0, len(metadata) - 1)
                    anno = metadata[new_idx]

        set_name = "MegaDepth"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "images": images,
            "image_paths": image_paths,
        }
        return batch