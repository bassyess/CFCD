#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import core.logging as logging
import datasets.transforms as transforms
import torch.utils.data
from core.config import cfg


logger = logging.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]]) # save as https://github.com/pytorch/pytorch/blob/master/caffe2/image/image_input_op.h#L276
# but in slow-fast, they are [0.225, 0.224, 0.229] https://github.com/facebookresearch/SlowFast/blob/52fb753f8f703b306896afc5613978db0c3c6695/slowfast/config/defaults.py#L524
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)


class DataSet(torch.utils.data.Dataset):
    """Common dataset."""

    def __init__(self, data_path, split, ret_path=False):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        logger.info("Constructing dataset from {}...".format(split))
        self._data_path, self._split, self._ret_path = data_path, split, ret_path
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # only train set use hard example mining
        self._path_gt_pred, self._index_gt_pred = {}, {}
        gt_pred_score = os.path.join(self._data_path,'landmark_scores.txt')
        if os.path.isfile(gt_pred_score) and "train" in self._split:
            with open(gt_pred_score,'r') as rd:
                for line in rd:
                    path, gt, pred, score=line.strip().split()
                    self._path_gt_pred[path]=(int(gt), int(pred))

        # Compile the split data path
        self._imdb, self._class_ids, self._class_indices = [], [], {}
        with open(os.path.join(self._data_path, self._split), "r") as fin:
            for index, line in enumerate(fin):
                im_path, cont_id = line.strip().split(" ")
                # im_path = os.path.join(self._data_path, im_dir)
                self._imdb.append({"im_path": im_path, "class": int(cont_id)})
                self._class_ids.append(int(cont_id))
                if int(cont_id) not in self._class_indices:
                    self._class_indices[int(cont_id)] = []
                self._class_indices[int(cont_id)].append(index)
                if self._path_gt_pred:
                    self._index_gt_pred[index] = self._path_gt_pred[im_path]
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(set(self._class_ids))))
        logger.info("Number of gt pred: {}".format(len(self._index_gt_pred)))

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        # Train and test setups differ
        train_size = cfg.TRAIN.IM_SIZE
        if "train" in self._split:
            # Scale and aspect ratio then horizontal flip
            im = transforms.random_sized_crop(im=im, size=train_size, area_frac=0.08)
            im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        else:
            # Scale and center crop
            assert train_size <=cfg.TEST.IM_SIZE
            im = transforms.scale(cfg.TEST.IM_SIZE, im) #保持比例，短边resize到目标大小
            im = transforms.center_crop(train_size, im)
        # HWC -> CHW
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # PCA jitter
        if "train" in self._split:
            im = transforms.lighting(im, 0.1, _EIG_VALS, _EIG_VECS)
        # Color normalization
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        # Load the image
        try:
            im = cv2.imread(self._imdb[index]["im_path"])
            im = im.astype(np.float32, copy=False)
        except:
            print('error: ', self._imdb[index]["im_path"])
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        if self._ret_path:
            return im, label, self._imdb[index]["im_path"]
        return im, label

    def __len__(self):
        return len(self._imdb)
