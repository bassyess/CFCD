# Copyright (C) 2021-2022 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
from torch import nn
import torch.nn.functional as F
import core.logging as logging
from core.config import cfg
logger = logging.get_logger(__name__)


def triplet_loss(x, label, margin=0.1):
    dim = x.size(0) 
    nq = torch.sum(label.data==-1).item() 
    S = cfg.DATA_LOADER.TUPLE_SIZE 
    assert(x.size(1)==S*nq)

    xa = x[:, label.data==-1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xp = x[:, label.data==1].permute(1,0).repeat(1,S-2).view((S-2)*nq,dim).permute(1,0)
    xn = x[:, label.data==0]

    dist_pos = torch.sum(torch.pow(xa - xp, 2), dim=0)
    dist_neg = torch.sum(torch.pow(xa - xn, 2), dim=0)

    return torch.sum(torch.clamp(dist_pos - dist_neg + margin, min=0))


def match_attn(query_feat, pos_feat, attention, LoweRatioTh=0.9):
    dist = torch.cdist(query_feat, pos_feat) 
    best1 = torch.argmin(dist, dim=1) 
    best2 = torch.argmin(dist, dim=0) 
    arange = torch.arange(best2.size(0), device=best2.device)
    reciprocal = best1[best2]==arange
    
    dist2 = dist.clone()
    dist2[best2,arange] = float('Inf') 
    dist2_second2 = torch.argmin(dist2, dim=0) 
    ratio1to2 = dist[best2,arange] / dist2[dist2_second2,arange] 
    valid = torch.logical_and(reciprocal, ratio1to2<=LoweRatioTh)
    _,i = attention.topk(attention.size(0)//2)
    mask = torch.zeros(attention.size(0), device=attention.device).scatter_(0,i,1)
    valid = torch.logical_and(valid, mask)
    pindices = torch.where(valid)[0]
    qindices = best2[pindices]
    
    return qindices, pindices


def get_nindices(query_feat, neg_feat):
    dist = torch.cdist(query_feat.detach(), neg_feat.detach())
    nearest_nindices = torch.argmin(dist, dim=1)
    return nearest_nindices


class TripletLoss(nn.Module):

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x, label):
        return triplet_loss(x, label, margin=self.margin)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')'


class SuperfeatureTripletLoss(nn.Module):
    
    def __init__(self, margin=1.1, weight=1.0):
        super().__init__()
        self.weight = weight
        self.criterion = TripletLoss(margin=margin)
        
    def forward(self, superfeatures_list, attention, target=None):
        """
        superfeatures_list is a list of tensor of size N x D containing the superfeatures for each image
        superfeatures_list:(TxNxD)
        target: 冗余参数，并没使用
        """
        if target is not None:
            assert target[0]==-1 and target[1]==1 and torch.all(target[2:]==0), "Only implemented for one tuple where the first element is the query, the second one the positive, and the rest are negatives"
        N = superfeatures_list[0].size(0)
        assert all(s.size(0)==N for s in superfeatures_list[1:]), "All images should have the same number of features"
        query_feat = F.normalize(superfeatures_list[0], dim=1)
        pos_feat = F.normalize(superfeatures_list[1], dim=1)
        neg_feat_list = [F.normalize(neg, dim=1) for neg in superfeatures_list[2:]]
        # perform matching 
        qindices, pindices = match_attn(query_feat, pos_feat, attention[1])
        if qindices.size(0)==0:
            return torch.sum(query_feat[:1,:1])*0.0 # for having a gradient that depends on the input to avoid torch error when using multiple processes
        # loss
        nneg = len(neg_feat_list)
        target = torch.Tensor(([-1, 1]+[0]*nneg) * len(qindices)).to(dtype=torch.int64, device=qindices.device)

        query_feat_selected = torch.index_select(query_feat,0,qindices)
        pos_feat_selected = torch.index_select(pos_feat,0,pindices)
        neg_feat_selected_list = [torch.index_select(neg_feat,0, get_nindices(query_feat_selected, neg_feat)) for neg_feat in neg_feat_list]
        catfeats = torch.cat([query_feat_selected.unsqueeze(1), pos_feat_selected.unsqueeze(1)] + \
                             [neg_feat_selected.unsqueeze(1) for neg_feat_selected in neg_feat_selected_list], dim=1) 
        catfeats = catfeats.view(-1, query_feat.size(1))

        loss = self.criterion(catfeats.T.contiguous(), target.detach())
        return loss * self.weight

    def __repr__(self):
        return "{:s}(margin={:g}, weight={:g})".format(self.__class__.__name__, self.criterion.margin, self.weight)
