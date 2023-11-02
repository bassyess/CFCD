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

""" Dolg models """

class DOLG(nn.Module):
    """ DOLG model """
    def __init__(self):
        super(DOLG, self).__init__()
        self.pool_l= nn.AdaptiveAvgPool2d((1, 1)) 
        self.pool_g = GeneralizedMeanPooling(norm=3.0) 
        self.fc_t = nn.Linear(cfg.MODEL.S4_DIM, cfg.MODEL.S3_DIM, bias=True)
        self.fc = nn.Linear(cfg.MODEL.S4_DIM, cfg.MODEL.HEADS.REDUCTION_DIM, bias=True)
        self.bn = nn.BatchNorm1d(cfg.MODEL.HEADS.REDUCTION_DIM)
        self.globalmodel = ResNet()
        self.localmodel = SpatialAttention2d(cfg.MODEL.S3_DIM, with_aspp=cfg.MODEL.WITH_MA)
        self.desc_cls = builders.get_head()(cfg.MODEL.HEADS.REDUCTION_DIM, cfg.MODEL.NUM_CLASSES)

    def forward(self, x, targets=None):
        """ Global and local orthogonal fusion """
        f3, f4 = self.globalmodel(x)
        fl, _ = self.localmodel(f3)
        
        fg_o = self.pool_g(f4)
        fg_o = fg_o.view(fg_o.size(0), cfg.MODEL.S4_DIM)
        
        fg = self.fc_t(fg_o)
        fg_norm = torch.norm(fg, p=2, dim=1)
        
        proj = torch.bmm(fg.unsqueeze(1), torch.flatten(fl, start_dim=2))
        proj = torch.bmm(fg.unsqueeze(2), proj).view(fl.size())
        proj = proj / (fg_norm * fg_norm).view(-1, 1, 1, 1)
        orth_comp = fl - proj

        fo = self.pool_l(orth_comp)
        fo = fo.view(fo.size(0), cfg.MODEL.S3_DIM)

        final_feat=torch.cat((fg, fo), 1)
        global_feature = self.fc(final_feat)
        global_feature = self.bn(global_feature)
        outdict = {"global_feature":global_feature}
        if targets is not None:
            global_logits = self.desc_cls(global_feature, targets)
            outdict["global_logits"] = global_logits
        return outdict


class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    '''
    def __init__(self, in_c, act_fn='relu', with_aspp=cfg.MODEL.WITH_MA):
        super(SpatialAttention2d, self).__init__()
        
        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(cfg.MODEL.S3_DIM)
        self.conv1 = nn.Conv2d(in_c, cfg.MODEL.S3_DIM, 1, 1)
        self.bn = nn.BatchNorm2d(cfg.MODEL.S3_DIM, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(cfg.MODEL.S3_DIM, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

        for conv in [self.conv1, self.conv2]: 
            conv.apply(net.init_weights)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        if self.with_aspp:
            x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        x = att * feature_map_norm
        return x, att_score
    
    def __repr__(self):
        return self.__class__.__name__


class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling Module 
    '''
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2d(in_c, 512, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2d(in_c, 512, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.ModuleList(self.aspp)

        self.im_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_c, 512, 1, 1),
                                     nn.ReLU())
        conv_after_dim = 512 * (len(self.aspp)+1)
        self.conv_after = nn.Sequential(nn.Conv2d(conv_after_dim, 1024, 1, 1), nn.ReLU())
        
        for dilation_conv in self.aspp:
            dilation_conv.apply(net.init_weights)
        for model in self.im_pool:
            if isinstance(model, nn.Conv2d):
                model.apply(net.init_weights)
        for model in self.conv_after:
            if isinstance(model, nn.Conv2d):
                model.apply(net.init_weights)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h,w), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = torch.cat(aspp_out, 1)
        x = self.conv_after(aspp_out)
        return x
