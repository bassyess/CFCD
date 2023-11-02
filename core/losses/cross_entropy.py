import torch.nn as nn
class CrossEntropyLoss(nn.Module):
    def __init__(self, weight= None, size_average=None, ignore_index = -100, reduce=None, reduction = 'mean'):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction)
    def forward(self, input, target):
        return self.loss(input["global_logits"], target)

