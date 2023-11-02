#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Written by feymanpriv

from numpy import finfo
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config import cfg

class Orthogonal(nn.Module):
    """ Orthogonal model """
    def __init__(self):
        super(Orthogonal, self).__init__()
        self.pool_l= nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(cfg.MODEL.HEADS.REDUCTION_DIM*2, cfg.MODEL.HEADS.REDUCTION_DIM, bias=True)
        self.bn = nn.BatchNorm1d(cfg.MODEL.HEADS.REDUCTION_DIM)

    def forward(self, fg, vlad_for_dolg):
        """ Global and local orthogonal fusion """
        fg = F.normalize(fg, p=2, dim=1)
        proj = torch.bmm(fg.unsqueeze(1), torch.flatten(vlad_for_dolg, start_dim=2))
        proj = torch.bmm(fg.unsqueeze(2), proj).view(vlad_for_dolg.size())
        orth_comp = vlad_for_dolg - proj 

        fo = self.pool_l(orth_comp)
        fo = fo.view(fo.size(0), cfg.MODEL.HEADS.REDUCTION_DIM)

        final_feat=torch.cat((fg, fo), 1)
        global_feature = self.fc(final_feat)
        global_feature = self.bn(global_feature)
        return global_feature

