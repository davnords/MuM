import gzip
import json
import os.path as osp
import logging

import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset
from PIL import UnidentifiedImageError

class HypersimDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        DATA_DIR: str = None,
        ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        sampling_weight: int = 1,
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

        self.invalid_sequence = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"Hypersim directory is {DATA_DIR}")

        self.DATA_DIR = DATA_DIR
        self.ANNOTATION_DIR = ANNOTATION_DIR

        total_frame_num = 0

        annotation_file = osp.join(
            self.ANNOTATION_DIR, f"hypersim.jgz"
        )

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {annotation_file}")

        for seq_name, seq_data in annotation.items():
            if len(seq_data) < min_num_images:
                continue
            if seq_name in self.invalid_sequence:
                continue
            total_frame_num += len(seq_data)
            self.data_store[seq_name] = seq_data

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Hypersim Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Hypersim Data length of training set: {len(self)}")
        logging.info(f"{status}: Hypersim Frame number: {total_frame_num}")

        del annotation

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:

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

        set_name = "Hypersim"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "images": images,
            "image_paths": image_paths,
        }
        return batch