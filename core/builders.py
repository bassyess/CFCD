#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

import torch
from core.config import cfg
from core.model.dolg_model import DOLG
from core.model.model_s1 import MODEL_S1
from core.model.model_s2 import MODEL_S2
from core.model.model_deploy import MODEL_DEPLOY
from core.losses.cross_entropy import CrossEntropyLoss
from core.losses.triplet_loss import TripletLoss
from datasets.commondataset import DataSet
from torch.utils.data.distributed import DistributedSampler
from datasets.distributed_tuple_sampler import DistributedTupleSampler
from datasets.distributed_tuple_hem_sampler import DistributedTupleHEMSampler, DistributedTuplePairSampler
from core.model.arcface import Arcface
from core.model.madacos import MadaCos
from core.model.fusion import Orthogonal

# Supported loss functions
_loss_funs = {"cross_entropy": CrossEntropyLoss, "triplet_loss": TripletLoss}
_models = {} # 当前并没有使用
_algos = {"DOLG":DOLG, "MODEL_S1":MODEL_S1, "MODEL_S2":MODEL_S2, "MODEL_DEPLOY":MODEL_DEPLOY}
_datasets = {"DataSet": DataSet}
_samplers = {"DistributedSampler": DistributedSampler, "DistributedTupleSampler":DistributedTupleSampler, "DistributedTupleHEMSampler":DistributedTupleHEMSampler, "DistributedTuplePairSampler":DistributedTuplePairSampler}
_heads = {"Arcface": Arcface, "MadaCos": MadaCos}
_fusionmodels = {"Orthogonal": Orthogonal}

def get_fusion():
    """Gets the fusionmodels class specified in the config."""
    err_str = "fusionmodels type '{}' not supported"
    assert cfg.MODEL.FUSIONMODELS.FUSIONMODEL in _fusionmodels.keys(), err_str.format(cfg.MODEL.FUSIONMODELS.FUSIONMODEL)
    return _fusionmodels[cfg.MODEL.FUSIONMODELS.FUSIONMODEL]

def get_head():
    """Gets the head class specified in the config."""
    err_str = "head type '{}' not supported"
    assert cfg.MODEL.HEADS.HEAD in _heads.keys(), err_str.format(cfg.MODEL.HEADS.HEAD)
    return _heads[cfg.MODEL.HEADS.HEAD]

def get_sampler():
    """Gets the sampler class specified in the config."""
    err_str = "sampler type '{}' not supported"
    assert cfg.DATA_LOADER.SAMPLER in _samplers.keys(), err_str.format(cfg.DATA_LOADER.SAMPLER)
    return _samplers[cfg.DATA_LOADER.SAMPLER]

def get_dataset():
    """Gets the dataset class specified in the config."""
    err_str = "dataset type '{}' not supported"
    assert cfg.DATA_LOADER.DATASET in _datasets.keys(), err_str.format(cfg.DATA_LOADER.DATASET)
    return _datasets[cfg.DATA_LOADER.DATASET]

def get_algo():
    """Gets the algo class specified in the config."""
    err_str = "Algo type '{}' not supported"
    assert cfg.MODEL.ALGO in _algos.keys(), err_str.format(cfg.MODEL.ALGO)
    return _algos[cfg.MODEL.ALGO]

def build_algo():
    """Build the model function."""
    return get_algo()()

def get_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]

def build_model():
    """Build the model function."""
    return get_model()()

def get_loss_fun():
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSSES.NAME in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSSES.NAME]


def build_loss_fun():
    """Build the loss function."""
    return get_loss_fun()()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor

