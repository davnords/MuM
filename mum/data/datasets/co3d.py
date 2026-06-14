# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import os
import logging
import io
from zipfile import ZipFile

import cv2
import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset


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


class Co3dDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_DIR: str = None,
        ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        sampling_weight: int = 1,
    ):
        """
        Initialize the Co3dDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            DATA_DIR (str): Directory path to CO3D data.
            ANNOTATION_DIR (str): Directory path to CO3D annotations.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If DATA_DIR or ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        
        self.duplicate_img = False
        self.load_depth = False

        if DATA_DIR is None or ANNOTATION_DIR is None:
            raise ValueError("Both DATA_DIR and ANNOTATION_DIR must be specified.")

        category = sorted(SEEN_CATEGORIES)

        self.sampling_weight = sampling_weight
        if split == "train":
            split_name_list = ["train"]
        elif split == "test":
            split_name_list = ["test"]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.invalid_sequence = [] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        self.DATA_DIR = DATA_DIR
        self.ANNOTATION_DIR = ANNOTATION_DIR

        total_frame_num = 0

        for c in category:
            for split_name in split_name_list:
                annotation_file = osp.join(
                    self.ANNOTATION_DIR, f"{c}_{split_name}.jgz"
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
                    if seq_name in self.invalid_sequence:
                        continue
                    total_frame_num += len(seq_data)
                    # print('Seq data:', seq_data)
                    self.data_store[seq_name] = seq_data

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Co3D Data size: {self.sequence_list_len}")
        logging.info(f"{status}: Co3D Data length of training set: {len(self)}")
        logging.info(f"{status}: Co3D Frame number: {total_frame_num}")

        del annotation

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
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
            filepath = anno["filepath"]

            image_path = osp.join(self.DATA_DIR, filepath)
            image = Image.open(image_path).convert("RGB")

            images.append(image)
            image_paths.append(image_path)

        set_name = "co3d"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": ids,
            "images": images,
            "image_paths": image_paths,
        }
        return batch