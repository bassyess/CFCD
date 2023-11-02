#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""

import os
import copy
import core.distributed as dist
import torch
from core.config import cfg
from pdb import set_trace as stop


# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"
# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not os.path.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))


def save_checkpoint(model, optimizer, epoch):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not dist.is_master_proc():
        return
    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    previous_checkpoint_file = get_checkpoint(epoch)
    if os.path.exists(previous_checkpoint_file) and (epoch % cfg.TRAIN.CHECKPOINT_PERIOD != 0):
        os.unlink(previous_checkpoint_file)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None, pretrained=True):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        ckpt_dict = checkpoint["model_state"]
    except KeyError:
        ckpt_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model.module if cfg.NUM_GPUS > 1 else model
    model_dict = ms.state_dict()
    if pretrained:
        ckpt_dict = {'globalmodel.'+ k : v for k, v in ckpt_dict.items()}
    loaded_dict = {k: v for k, v in ckpt_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(loaded_dict) == len(ckpt_dict) and len(loaded_dict)==len(model_dict):
        print('All params loaded! Same model!')
    else:
        print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(ckpt_dict)))
        print('{} pretrain keys load successfully.'.format(len(loaded_dict)))
        not_loaded_keys = [k for k in ckpt_dict.keys() if k not in loaded_dict.keys()]
        if len(not_loaded_keys) > 0:
            print('not_loaded_keys', ('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
        request_loaded_keys = [k for k in model_dict.keys()  if k not in loaded_dict.keys()]
        if len(request_loaded_keys) > 0:
            print('request_loaded_keys', ('%s, ' * (len(request_loaded_keys) - 1) + '%s') % tuple(request_loaded_keys))
    model_dict.update(loaded_dict)
    ms.load_state_dict(model_dict)
    #ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    #return checkpoint["epoch"]
    return checkpoint
