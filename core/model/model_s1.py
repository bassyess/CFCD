import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import core.net as net
from core import builders
from core.config import cfg
from core.model.resnet import ResNet, ResHead
from core.model.resnet import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from pdb import set_trace as stop
""" ResNet models """

class MODEL_S1(nn.Module):
    """ model s1"""
    def __init__(self):
        super(MODEL_S1, self).__init__()
        self.pool_g = GeneralizedMeanPooling(norm=3.0)
        self.fc = nn.Linear(cfg.MODEL.S4_DIM, cfg.MODEL.HEADS.REDUCTION_DIM, bias=True)
        self.bn = nn.BatchNorm1d(cfg.MODEL.HEADS.REDUCTION_DIM)
        self.globalmodel = ResNet()
        self.desc_cls = builders.get_head()(cfg.MODEL.HEADS.REDUCTION_DIM, cfg.MODEL.NUM_CLASSES)

    def forward(self, x, targets=None):
        f3, f4 = self.globalmodel(x)
        fg_o = self.pool_g(f4)
        fg_o = fg_o.view(fg_o.size(0), cfg.MODEL.S4_DIM)
        
        global_feature = self.fc(fg_o)
        global_feature = self.bn(global_feature)
        outdict = {"global_feature":global_feature}
        if targets is not None:
            global_logits = self.desc_cls(global_feature, targets)
            outdict["global_logits"] = global_logits
        return outdict
