import torch,math
from torch import nn
from torch import autograd
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import Parameter
import core.logging as logging
logger = logging.get_logger(__name__)
from core.config import cfg
from pdb import set_trace as stop


class MadaCosLayerFunction(autograd.Function):

    @staticmethod
    def forward(ctx, logit, label, training=True, method_B='default', alpha=0.1):
        assert logit.size(0) == label.size(0)
        assert alpha > 0 and alpha < 1

        if cfg.NUM_GPUS == 1:
            world_size = 1
        else:
            world_size = dist.get_world_size()

        onehot = torch.zeros_like(logit)
        onehot.scatter_(1, label.view(-1,1).long(), 1)

        target_logit = torch.sum(torch.where(onehot > 0, logit, torch.zeros_like(logit)), dim=1)
        if world_size > 1:
            gather_list = [torch.empty_like(target_logit) for _ in range(world_size)]
            dist.all_gather(gather_list, target_logit) 
            target_logit = torch.cat(gather_list, dim = 0)

        logit_median, index = torch.median(target_logit, dim=0, keepdim=True)
        scale = math.log(9999999 * (1.-alpha) / alpha) / (1 - logit_median)
        if scale.item() < 32:
            scale.fill_(32.)
        B = torch.sum(torch.where(onehot < 1, torch.exp(logit * scale), torch.zeros_like(logit)), dim=1)
        
        if world_size > 1:
            gather_list = [torch.empty_like(B) for _ in range(world_size)]
            dist.all_gather(gather_list, B) 
            B = torch.cat(gather_list, dim = 0)
        
        if method_B=='median':
            B_avg = torch.median(B)
        elif method_B=='mean':
            B_avg = torch.mean(B)
        else:
            B_avg = B[index]
        margin = logit_median - torch.log(B_avg * alpha / (1.-alpha)) / scale
        if margin.item() < 0 or training==False:
            margin.fill_(0)
        m3 = margin.item()

        ctx.save_for_backward( scale)
        logit = logit * scale
        if m3!=0:
            new_logit = logit - scale * m3
            logit = torch.where(onehot > 0, new_logit, logit)

        return logit, margin, scale, logit_median, B_avg

    @staticmethod
    def backward(ctx, grad_logit, grad_margin, grad_scale, grad_logit_median, grad_B_avg):
        scale, = ctx.saved_tensors
        return grad_logit * scale, None, None, None, None

class MadaCosLayer(nn.Module):
    def __init__(self, method_B='default', alpha=0.1):
        super(MadaCosLayer, self).__init__()
        self.method_B = method_B
        self.alpha = alpha

    def forward(self, logit, label):
        return MadaCosLayerFunction.apply(logit, label, self.training, self.method_B, self.alpha)

# def madacos_layer(logit, label):
#     return MadaCosLayerFunction.apply(logit, label, method_B ='default', alpha=0.1)

class MadaCos(nn.Module):
    """ MadaCosLoss """
    def __init__(self, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self.alpha = cfg.MODEL.HEADS.ALPHA
        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        self.madacos_layer = MadaCosLayer(alpha=self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features, targets):
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        logit, margin, scale, logit_median, B_avg = self.madacos_layer(cos_theta, targets)
        # stats = {'logit_median':logit_median.item(), 'B_avg':B_avg.item(), 'scale':scale.item(), 'margin':margin.item()}
        # logger.info(logging.dump_log_data(stats, "{}_iter".format('train' if self.training else 'test')))
        return logit

    def extra_repr(self):
        return 'in_features={}, num_classes={}'.format(self.in_feat, self._num_classes)
