#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys, os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import init_path
import numpy as np
import cv2
import pickle
from scipy.io import savemat 

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import core.config as config
import core.builders as builders

from process import preprocess
from dataset import configdataset
from util import walkfile, l2_norm

""" common settings """
SCALE_LIST = [0.3535, 0.5, 0.7071, 1.0, 1.4142]
# SCALE_LIST = [0.7071, 1.0, 1.4142]


def setup_model(ckpt_path):
    # Creator = eval(cfg.MODEL.ALGO)
    # model = Creator()
    model = builders.build_algo()
    print(model)
    load_checkpoint(ckpt_path, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def extract(img, model):
    globalfeature = None
    for s in SCALE_LIST:
        im = preprocess(img.copy(), s)
        input_data = np.asarray([im], dtype=np.float32)
        input_data = torch.from_numpy(input_data)
        print(input_data.shape)
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        outdict = model(input_data)
        global_feature = outdict["global_feature"]
        global_feature = F.normalize(global_feature, p=2, dim=1)
        if globalfeature is None:
            globalfeature = global_feature.cpu().detach().numpy()
        else:
            globalfeature += global_feature.cpu().detach().numpy()
    global_feature = globalfeature / len(SCALE_LIST)
    global_feature = l2_norm(global_feature)
    return global_feature


def main(ckpt, spath, data_cfg, opath):
    extractor = extract
    with torch.no_grad():
        model = setup_model(ckpt_path=ckpt)
        feadic = {}
        for index, imgfile in enumerate(walkfile(spath)):
            ext = os.path.splitext(imgfile)[-1]
            name = os.path.basename(imgfile).split('.')[0]
            if name not in data_cfg['qimlist'] and name not in data_cfg['imlist']:
                continue
            if ext.lower() in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
                im = cv2.imread(imgfile)
                print(index, imgfile, end=' ')
                print(im.shape, end='')
                if name in data_cfg['qimlist']:
                    pos = data_cfg['qimlist'].index(name)
                    x1, y1, x2, y2 = map(int, data_cfg['gnd'][pos]['bbx'])
                    cropped_im = im[y1:y2, x1:x2] #crop query image
                    im = cropped_im
                    print('->',im.shape)
                else:
                    print("")
                im = im.astype(np.float32, copy=False)
                data = extractor(im, model)
                print(data.shape)
                feadic[name] = data
    with open(opath, "wb") as fout:
        print('save to {}'.format(opath), len(feadic))
        pickle.dump(feadic, fout, protocol=2)


def main_multicard(ckpt, spath, opath, cutno, total_num):
    """multi processes for extracting 1M distractors features"""
    extractor = extract
    with torch.no_grad():
        model = setup_model(ckpt_path=ckpt)
        feadic = {'X':[]}
        for index, imgfile in enumerate(walkfile(spath)):
            if index % total_num != cutno - 1:
                continue
            ext = os.path.splitext(imgfile)[-1]
            name = os.path.basename(imgfile)
            print(index, imgfile)
            if ext.lower() in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
                im = cv2.imread(imgfile)
                h, w = im.shape[:2]
                im = im.astype(np.float32, copy=False)
                data =  extractor(im, model)
                print(data.shape)
                feadic['X'].append(data)
        print('save to {}'.format(opath),len(feadic['X']))
        savemat(opath,feadic)


def load_checkpoint(checkpoint_file, model):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        ckpt_dict = checkpoint["model_state"]
    except KeyError:
        ckpt_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in ckpt_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(loaded_dict) == len(ckpt_dict) and len(loaded_dict)==len(model_dict):
        print('All params loaded! Same model!')
    else:
        print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(ckpt_dict)))
        print('{} pretrain keys load successfully.'.format(len(loaded_dict)))
        not_loaded_keys = [k for k in ckpt_dict.keys()  if k not in loaded_dict.keys()]
        print('not_loaded_keys', ('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
        request_loaded_keys = [k for k in model_dict.keys()  if k not in loaded_dict.keys()]
        print('request_loaded_keys', ('%s, ' * (len(request_loaded_keys) - 1) + '%s') % tuple(request_loaded_keys))
    model.load_state_dict(loaded_dict)
    return checkpoint


if __name__ == '__main__':
    print(sys.argv)
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    assert cfg.INFER.TOTAL_NUM > 0 and cfg.INFER.CUT_NUM <=  cfg.INFER.TOTAL_NUM
    ckpt_path = cfg.INFER.CKPT
    data_path = cfg.INFER.DIR
    test_dataset = cfg.INFER.TEST
    print(cfg)
    if test_dataset == 'roxford5k' or test_dataset == 'rparis6k':
        DATA_DIR = '{}/{}'.format(data_path, test_dataset)
        data_cfg = configdataset(test_dataset, DATA_DIR)
        INFER_DIR = os.path.join(DATA_DIR, 'jpg')
        main(ckpt_path, INFER_DIR, data_cfg, os.path.splitext(ckpt_path)[0]+'_'+test_dataset+'.pickle')
    elif test_dataset == 'revisitop1m':
        DATA_DIR = '{}/{}'.format(data_path, test_dataset)
        INFER_DIR = os.path.join(DATA_DIR, 'jpg')
        main_multicard(ckpt_path, INFER_DIR, os.path.splitext(ckpt_path)[0]+'_'+test_dataset+'_{}.mat'.format(cfg.INFER.CUT_NUM), cfg.INFER.CUT_NUM, cfg.INFER.TOTAL_NUM)
    