#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import os

import numpy as np
import core.benchmark as benchmark
import core.builders as builders
import core.checkpoint as checkpoint
import core.config as config
import core.distributed as dist
import core.logging as logging
import core.meters as meters
import core.net as net
import core.optimizer as optim
import datasets.loader as loader
import torch
from core.config import cfg


logger = logging.get_logger(__name__)



def setup_env(training=True):
    """Sets up environment for training or testing."""
    if dist.is_master_proc() and training:
        # Ensure that the output dir exists
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        # Save the config
        config.dump_cfg()
    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    # Creator = eval(cfg.MODEL.ALGO)
    model = builders.build_algo()
    logger.info("Model:\n{}".format(model))
    # Log model complexity
    #logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, find_unused_parameters=True
        )
        # Set complexity function to be module's complexity function
        #model.complexity = model.module.complexity
    return model


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        outdict = model(inputs, labels)
        # Compute the loss
        desc_loss = loss_fun(outdict, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        # Freeze localmodel
        desc_loss.backward()
        # update params
        optimizer.step()

        # Compute the errors
        global_logits = outdict["global_logits"]
        desc_top1_err, desc_top5_err = meters.topk_errors(global_logits, labels, [1, 5])
        desc_loss, desc_top1_err, desc_top5_err = dist.scaled_all_reduce([desc_loss, desc_top1_err, desc_top5_err])
        desc_loss, desc_top1_err, desc_top5_err = desc_loss.item(), desc_top1_err.item(), desc_top5_err.item()

        train_meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        train_meter.update_stats(desc_top1_err, desc_top5_err, desc_loss, lr, mb_size)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def test_epoch(test_loader, model, loss_fun, test_meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        outdict = model(inputs, labels)
        # Compute the loss
        desc_loss = loss_fun(outdict, labels)
        # Compute the errors
        global_logits = outdict["global_logits"]
        top1_err, top5_err = meters.topk_errors(global_logits, labels, [1, 5])
        desc_loss, top1_err, top5_err = dist.scaled_all_reduce([desc_loss, top1_err, top5_err])
        desc_loss, top1_err, top5_err = desc_loss.item(), top1_err.item(), top5_err.item()

        test_meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        test_meter.update_stats(top1_err, top5_err, desc_loss, mb_size)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    logger.info("Loss: {}".format(loss_fun))
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer, pretrained=False)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = int(checkpoint_epoch['epoch']) + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model, pretrained=cfg.TRAIN.PRETRAINED)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    # Compute model and loader timings
    #if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        #benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        # if (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
        checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
        logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        next_epoch = cur_epoch + 1
        if next_epoch % cfg.TRAIN.EVAL_PERIOD == 0 or next_epoch == cfg.OPTIM.MAX_EPOCH:
            test_epoch(test_loader, model, loss_fun, test_meter, cur_epoch)


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env(training=False)
    # Construct the model
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model, pretrained=False)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, loss_fun, test_meter, 0)

def infer_model():
    """inference a trained model."""
    # Setup training/testing environment
    setup_env(training=False)
    # Construct the model
    model = setup_model()
    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model, pretrained=False)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    infer_loader = loader.construct_infer_loader()
    # Evaluate the model
    infer_epoch(infer_loader, model)

@torch.no_grad()
def infer_epoch(infer_loader, model):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    total_batch = len(infer_loader)
    if cfg.NUM_GPUS == 1:
        rank = 'a'
    else:
        rank = torch.distributed.get_rank()
    with open(os.path.splitext(cfg.TEST.WEIGHTS)[0]+'_{}.info'.format(rank),'w') as wd:
        for cur_iter, (inputs, labels, paths) in enumerate(infer_loader):
            # Transfer the data to the current GPU device
            inputs, labels_g = inputs.cuda(), labels.cuda(non_blocking=True)
            # Compute the predictions
            outdict = model(inputs, labels_g)
            # Compute the errors
            
            global_logits = outdict["global_logits"]
            output = torch.nn.functional.softmax(global_logits,dim=1)
            scores, indexs = output.topk(1, 1, True, True)
            for path, index, score, label in zip(paths,indexs,scores, labels):
                print('{}/{}'.format(cur_iter, total_batch), path, label.item(), index.item(), score.item())
                wd.write('{} {} {} {}\n'.format(path, label.item(), index.item(), score.item()))

def time_model():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env(training=False)
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()
    # Compute model and loader timings
    #benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
