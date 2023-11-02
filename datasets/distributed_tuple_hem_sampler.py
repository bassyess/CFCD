import math
from typing import  Optional
from torch.utils.data import Sampler
from torch.utils.data import Dataset
import torch.distributed as dist
from core.config import cfg
import random, copy
class DistributedTupleHEMSampler(Sampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.hard_example = {}
        self.class_indices = dataset._class_indices
        self.index_gt_pred = dataset._index_gt_pred
        self.negative_num = cfg.DATA_LOADER.TUPLE_SIZE - 2
        self.drop_last = drop_last 
        self.lcm_replicas_tuplesize = int(self.num_replicas * cfg.DATA_LOADER.TUPLE_SIZE)
        # If the dataset length is evenly divisible by # of replicas, then there is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.lcm_replicas_tuplesize != 0:
            # Split to nearest available length that is evenly divisible. This is to ensure each rank receives the same amount of data when using this Sampler.
            self.num_samples = math.ceil( (len(self.dataset) - self.lcm_replicas_tuplesize) / self.lcm_replicas_tuplesize)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.lcm_replicas_tuplesize)
        self.total_size = self.num_samples * self.lcm_replicas_tuplesize
        self.num_samples = int(self.total_size / self.num_replicas)
        assert self.num_samples % cfg.DATA_LOADER.TUPLE_SIZE == 0, "split to each gpu"
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        candidate_anchor_class = []
        candidate_class_indices = copy.deepcopy(self.class_indices)
        for c,v in candidate_class_indices.items():
            random.shuffle(v)
            if len(v)>=2:
                candidate_anchor_class.append(c)
        if self.index_gt_pred:
            class_TP_FP = {}
            for index,(gt, pred) in self.index_gt_pred.items():
                if gt not in class_TP_FP:
                    class_TP_FP[gt]={'TP':[],'FP':[]}
                if pred not in class_TP_FP:
                    class_TP_FP[pred]={'TP':[],'FP':[]}
                if gt==pred:
                    class_TP_FP[gt]['TP'].append(index)
                else:
                    class_TP_FP[pred]['FP'].append(index)
            for k,d in class_TP_FP.items():
                for t,v in d.items():
                    random.shuffle(v)
        indices = []
        while len(indices)<self.total_size:
            # reset candidate_anchor_class candidate_class_indices
            if len(candidate_anchor_class)==0:
                candidate_anchor_class = []
                candidate_class_indices = copy.deepcopy(self.class_indices)
                for c,v in candidate_class_indices.items():
                    random.shuffle(v)
                    if len(v)>=2:
                        candidate_anchor_class.append(c)
                if self.index_gt_pred:
                    class_TP_FP = {}
                    for index,(gt, pred) in self.index_gt_pred.items():
                        if gt not in class_TP_FP:
                            class_TP_FP[gt]={'TP':[],'FP':[]}
                        if pred not in class_TP_FP:
                            class_TP_FP[pred]={'TP':[],'FP':[]}
                        if gt==pred:
                            class_TP_FP[gt]['TP'].append(index)
                        else:
                            class_TP_FP[pred]['FP'].append(index)
                    for k,d in class_TP_FP.items():
                        for t,v in d.items():
                            random.shuffle(v)

            # select anchor class
            c = random.choice(candidate_anchor_class)
            # select anchor
            a = candidate_class_indices[c].pop()
            # select positive 
            if self.index_gt_pred:
                a_is_tp = True
                if a in class_TP_FP[c]['TP']: 
                    a_is_tp = True
                    class_TP_FP[c]['TP'].remove(a)
                    p = candidate_class_indices[c].pop() 
                    if p in class_TP_FP[c]['TP']:
                        class_TP_FP[c]['TP'].remove(p)
                    else:
                        class_TP_FP[self.index_gt_pred[p][1]]['FP'].remove(p)
                else:
                    a_is_tp = False
                    class_TP_FP[self.index_gt_pred[a][1]]['FP'].remove(a)
                    if class_TP_FP[c]['TP']: 
                        p = class_TP_FP[c]['TP'].pop()
                        candidate_class_indices[c].remove(p)
                    else:
                        p = candidate_class_indices[c].pop() 
                        class_TP_FP[self.index_gt_pred[p][1]]['FP'].remove(p)
            else:
                p = candidate_class_indices[c].pop()
            length_c = len(candidate_class_indices[c])
            if length_c == 0:
                del candidate_class_indices[c]
                candidate_anchor_class.remove(c)
            elif length_c < 2:
                candidate_anchor_class.remove(c)
            # select negative 
            ns = []
            if self.index_gt_pred:
                for i in range(self.negative_num):
                    nc = None
                    if a_is_tp:
                        if random.random()<0.5 and class_TP_FP[c]['FP']:
                            n = class_TP_FP[c]['FP'].pop()
                            ns.append(n)
                            nc = self.index_gt_pred[n][0]
                            candidate_class_indices[nc].remove(n)
                    else:
                        if random.random()<0.33 and class_TP_FP[c]['FP']:
                            n = class_TP_FP[c]['FP'].pop()
                            ns.append(n)
                            nc = self.index_gt_pred[n][0]
                            candidate_class_indices[nc].remove(n)
                        elif random.random()<0.5 and self.index_gt_pred[a][1] in candidate_class_indices: 
                            nc = self.index_gt_pred[a][1]
                            n = candidate_class_indices[nc].pop()
                            ns.append(n)
                            if n in class_TP_FP[nc]['TP']:
                                class_TP_FP[nc]['TP'].remove(n)
                            else:
                                class_TP_FP[self.index_gt_pred[n][1]]['FP'].remove(n)
                    if nc is not None: #nc could be 0
                        length_n = len(candidate_class_indices[nc])
                        if length_n == 0:
                            del candidate_class_indices[nc]
                        if length_n < 2 and nc in candidate_anchor_class:
                                candidate_anchor_class.remove(nc)
            # select negative class
            candidate_negative_class = list(candidate_class_indices.keys())
            if length_c:
                candidate_negative_class.remove(c)
            # random.shuffle(candidate_negative_class)
            if len(candidate_negative_class) > self.negative_num:
                candidate_negative_class=random.sample(candidate_negative_class, self.negative_num)

            while len(ns) < self.negative_num and len(candidate_negative_class):
                candidate_negative_class_update = []
                for nc in candidate_negative_class:
                    n = candidate_class_indices[nc].pop()
                    ns.append(n)
                    if self.index_gt_pred:
                        if n in class_TP_FP[nc]['TP']:
                            class_TP_FP[nc]['TP'].remove(n)
                        else:
                            class_TP_FP[self.index_gt_pred[n][1]]['FP'].remove(n)
                    length_n = len(candidate_class_indices[nc])
                    if length_n:
                        candidate_negative_class_update.append(nc)
                    else:
                        del candidate_class_indices[nc]
                    if length_n < 2 and nc in candidate_anchor_class:
                        candidate_anchor_class.remove(nc)
                    if len(ns)>=self.negative_num:
                        break
                candidate_negative_class = candidate_negative_class_update
            if len(ns)!=self.negative_num:
                candidate_anchor_class = []
                candidate_class_indices = {}
                continue
            indices.append(a)
            indices.append(p)
            indices.extend(ns)
        # subsample
        assert len(indices) == self.total_size
        offset = self.num_samples * self.rank
        indices = indices[offset:offset+self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, hard_example = {}) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.hard_example = hard_example
        self.epoch = epoch


class DistributedTuplePairSampler(Sampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.hard_example = {} 
        self.class_indices = dataset._class_indices
        self.index_gt_pred = dataset._index_gt_pred
        # self.negative_num = cfg.DATA_LOADER.TUPLE_SIZE - 2
        self.drop_last = drop_last 
        self.lcm_replicas_tuplesize = int(self.num_replicas * cfg.DATA_LOADER.TUPLE_SIZE)
        # If the dataset length is evenly divisible by # of replicas, then there is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.lcm_replicas_tuplesize != 0:
            # Split to nearest available length that is evenly divisible. This is to ensure each rank receives the same amount of data when using this Sampler.
            self.num_samples = math.ceil( (len(self.dataset) - self.lcm_replicas_tuplesize) / self.lcm_replicas_tuplesize)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.lcm_replicas_tuplesize)
        self.total_size = self.num_samples * self.lcm_replicas_tuplesize
        self.num_samples = int(self.total_size / self.num_replicas)
        assert self.num_samples % cfg.DATA_LOADER.TUPLE_SIZE == 0, "split to each gpu"
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        random.seed(self.seed + self.epoch)
        candidate_anchor_class = []
        candidate_class_indices = copy.deepcopy(self.class_indices)
        for c,v in candidate_class_indices.items():
            random.shuffle(v)
            if len(v)>=2:
                candidate_anchor_class.append(c)
        if self.index_gt_pred:
            class_TP_FP = {}
            for index,(gt, pred) in self.index_gt_pred.items():
                if gt not in class_TP_FP:
                    class_TP_FP[gt]={'TP':[],'FP':[]}
                if pred not in class_TP_FP:
                    class_TP_FP[pred]={'TP':[],'FP':[]}
                if gt==pred:
                    class_TP_FP[gt]['TP'].append(index)
                else:
                    class_TP_FP[pred]['FP'].append(index)
            for k,d in class_TP_FP.items():
                for t,v in d.items():
                    random.shuffle(v)
        indices = []
        while len(indices)<self.total_size:
            # reset candidate_anchor_class candidate_class_indices
            if len(candidate_anchor_class)==0:
                candidate_anchor_class = []
                candidate_class_indices = copy.deepcopy(self.class_indices)
                for c,v in candidate_class_indices.items():
                    random.shuffle(v)
                    if len(v)>=2:
                        candidate_anchor_class.append(c)
                if self.index_gt_pred:
                    class_TP_FP = {}
                    for index,(gt, pred) in self.index_gt_pred.items():
                        if gt not in class_TP_FP:
                            class_TP_FP[gt]={'TP':[],'FP':[]}
                        if pred not in class_TP_FP:
                            class_TP_FP[pred]={'TP':[],'FP':[]}
                        if gt==pred:
                            class_TP_FP[gt]['TP'].append(index)
                        else:
                            class_TP_FP[pred]['FP'].append(index)
                    for k,d in class_TP_FP.items():
                        for t,v in d.items():
                            random.shuffle(v)

            # select anchor class
            c = random.choice(candidate_anchor_class)
            # select anchor
            a = candidate_class_indices[c].pop()
            # select positive 
            if self.index_gt_pred:
                a_is_tp = True
                if a in class_TP_FP[c]['TP']:
                    a_is_tp = True
                    class_TP_FP[c]['TP'].remove(a)
                    p = candidate_class_indices[c].pop() 
                    if p in class_TP_FP[c]['TP']:
                        class_TP_FP[c]['TP'].remove(p)
                    else:
                        class_TP_FP[self.index_gt_pred[p][1]]['FP'].remove(p)
                else:
                    a_is_tp = False
                    class_TP_FP[self.index_gt_pred[a][1]]['FP'].remove(a)
                    if class_TP_FP[c]['TP']: 
                        p = class_TP_FP[c]['TP'].pop()
                        candidate_class_indices[c].remove(p)
                    else:
                        p = candidate_class_indices[c].pop()
                        class_TP_FP[self.index_gt_pred[p][1]]['FP'].remove(p)
            else:
                p = candidate_class_indices[c].pop()
            length_c = len(candidate_class_indices[c])
            if length_c == 0:
                del candidate_class_indices[c]
                candidate_anchor_class.remove(c)
            elif length_c < 2:
                candidate_anchor_class.remove(c)
            
            indices.append(a)
            indices.append(p)

        # subsample
        assert len(indices) == self.total_size
        offset = self.num_samples * self.rank
        indices = indices[offset:offset+self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int, hard_example = {}) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.hard_example = hard_example
        self.epoch = epoch