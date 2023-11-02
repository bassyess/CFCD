from cv2 import destroyWindow
import torch.nn as nn
from core.losses.super_loss import SuperfeatureTripletLoss
from core.config import cfg
import core.logging as logging
logger = logging.get_logger(__name__)

class TripletLoss(nn.Module):
    '''
        input: {"global_logits":global_logits,  "res4":res4}
    '''
    def __init__(self, margin=0.1, superfeature_loss_weight=1.0):
        super().__init__()
        self.margin = cfg.MODEL.LOSSES.MARGIN
        self.superfeature_loss_weight = cfg.MODEL.LOSSES.LAMDA
        self.triplet_loss = SuperfeatureTripletLoss(margin=margin, weight=superfeature_loss_weight)
        self.softmax_loss = nn.CrossEntropyLoss()
        
    def forward(self, in_dict, target):
        global_logits = in_dict["global_logits"]
        class_loss = self.softmax_loss(global_logits, target)
        
        res4 = in_dict["res4"].flatten(start_dim=2).permute(0,2,1) #Bx2048xHxW->Bx2048xHW->BxHWx2048
        attention = in_dict["attention"]
        
        assert res4.size(0) % cfg.DATA_LOADER.TUPLE_SIZE == 0, "batch size in single gpu must be divisible by the tuple_size"
        K,D = res4.shape[1:]
        res4 = res4.reshape(-1, cfg.DATA_LOADER.TUPLE_SIZE, K, D)
        attention = attention.reshape(-1, cfg.DATA_LOADER.TUPLE_SIZE, K)
        triplet_loss = 0.0
        for res4_t, att in zip(res4, attention):
            triplet_loss+=self.triplet_loss(res4_t, att)
        # stats = {'triplet_loss':triplet_loss.item(), 'class_loss': class_loss.item()}
        # logger.info(logging.dump_log_data(stats, "iter"))
        
        return class_loss + triplet_loss 
        
    def __repr__(self):
        return "{:s}(margin={:g}, superfeature_loss_weight={:g})" \
        .format(self.__class__.__name__, self.margin, self.superfeature_loss_weight)
