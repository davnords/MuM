# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os.path as osp
import logging

import random
import numpy as np


from data.dataset_util import *
from data.base_dataset import BaseDataset

Image.MAX_IMAGE_PIXELS = None


class MegaScenesDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        DATA_DIR: str = None,
        ANNOTATION_DIR: str = None,
        min_num_images: int = 24,
        sampling_weight: int = 2,
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

        self.sampling_weight = sampling_weight

        self.invalid_sequence = [
            "265_483_0", 
            "219_164_1", 
            "178_996_0", 
            "027_783_1",
            "104_860_0",
            "013_891_3",
            "219_147_0",
            "149_108_4",
            "219_164_0",
            "364_544_0",
            "171_565_3",
            "025_646_1",
            "216_140_0",
            "104_826_0",
            "364_059_1",
            "302_353_0",
            "027_783_0",
            "103_880_0",
            "003_846_0",
            "217_292_0",
            "027_983_0",
            "195_470_0",
            "149_108_0",
            "178_584_0",
            "113_735_0",
            "027_783_2",
            "219_164_2",
            "185_848_1",
            "149_108_12",
            "177_800_1",
            "003_081_0",
            "140_463_0",
            "128_018_1",
            "218_952_0",
            "437_840_0",
            "003_024_1",
            "024_944_3",
            "399_146_0",
            "219_784_5",
            "186_591_0",
            "083_850_0",
            "215_905_0",
            "197_716_0",
            "121_402_4",
            "174_752_4",
            "013_924_0",
            "195_317_0",
            ] # set any invalid sequence names here


        self.category_map = {}
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"MegaScenes directory is {DATA_DIR}")

        self.DATA_DIR = DATA_DIR
        self.ANNOTATION_DIR = ANNOTATION_DIR

        total_frame_num = 0

        annotation_file = osp.join(
            self.ANNOTATION_DIR, f"megascenes.jgz"
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
        logging.info(f"{status}: MegaScenes Data size: {self.sequence_list_len}")
        logging.info(f"{status}: MegaScenes Data length of training set: {len(self)}")
        logging.info(f"{status}: MegaScenes Frame number: {total_frame_num}")

        del annotation

    def _load_image_safely(self, image_path):
        """
        Safely load an image with fallback options.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            PIL.Image or None: Loaded image or None if failed
        """
        try:
            full_path = osp.join(self.DATA_DIR, image_path)
            image = Image.open(full_path).convert("RGB")
            # Verify the image can be loaded properly
            image.load()
            return image
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            return None
        except Exception as e:
            logging.warning(f"Unexpected error loading image {image_path}: {e}")
            return None
        
    def _try_load_sequence(self, seq_name, img_per_seq, ids=None):
        """
        Try to load a complete sequence. Returns None if any image fails.
        """
        try:
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
                full_path = self.DATA_DIR + "/" + filepath
                
                image = Image.open(full_path).convert("RGB")
                image.load()  # Verify image can be loaded
                
                images.append(image)
                image_paths.append(filepath)

            return {
                "ids": ids,
                "images": images,
                "image_paths": image_paths,
            }
        except Exception as e:
            logging.warning(f"Failed to load sequence {seq_name}: {e}")
            return None

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence with fallback to other sequences.
        """
        original_seq_name = seq_name
        
        max_attempts = 5
        for attempt in range(max_attempts):
            # Select sequence
            if seq_name is None or attempt > 0:
                # Get available sequences (exclude invalid ones)
                available_sequences = [s for s in self.sequence_list if s not in self.invalid_sequence]
                
                if not available_sequences:
                    raise RuntimeError("No valid sequences available")
                    
                if self.inside_random or seq_name is None:
                    seq_index = random.randint(0, len(available_sequences) - 1)
                    seq_name = available_sequences[seq_index]
                else:
                    seq_name = available_sequences[seq_index] if seq_index < len(available_sequences) else available_sequences[0]

            # Try to load this sequence
            result = self._try_load_sequence(seq_name, img_per_seq, ids)
            
            if result is not None:
                # Success! Return the batch
                set_name = "megascenes"
                batch = {
                    "seq_name": set_name + "_" + seq_name,
                    **result
                }
                return batch
            else:
                # Reset seq_name for next attempt (unless we want to keep trying the original)
                if original_seq_name is not None and attempt == 0:
                    # First attempt with specific seq_name failed, now try random ones
                    seq_name = None
                else:
                    seq_name = None

        raise RuntimeError(f"Failed to get valid sequence data after {max_attempts} attempts")