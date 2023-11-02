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
""" ResNet models """

class MODEL_DEPLOY(nn.Module):
    """ MODEL Deploy """
    def __init__(self):
        super(MODEL_DEPLOY, self).__init__()
        self.pool_g = GeneralizedMeanPooling(norm=3.0) 
        self.fc = nn.Linear(cfg.MODEL.S4_DIM, cfg.MODEL.HEADS.REDUCTION_DIM, bias=True)
        self.bn = nn.BatchNorm1d(cfg.MODEL.HEADS.REDUCTION_DIM)
        self.globalmodel = ResNet()
        
    def forward(self, x):
        f3, f4 = self.globalmodel(x)
        fg_o = self.pool_g(f4)
        fg_o = fg_o.view(fg_o.size(0), cfg.MODEL.S4_DIM)
        
        global_feature = self.fc(fg_o)
        global_feature = self.bn(global_feature)
        global_feature = F.normalize(global_feature, p=2, dim=1)
        
        return global_feature
