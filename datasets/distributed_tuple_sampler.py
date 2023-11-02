import math
from typing import  Optional
from torch.utils.data import Sampler
from torch.utils.data import Dataset
import torch.distributed as dist
from core.config import cfg
import random, copy
class DistributedTupleSampler(Sampler):
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

            # select anchor class
            c = random.choice(candidate_anchor_class)
            # select anchor
            a = candidate_class_indices[c].pop()
            # select positive 
            p = candidate_class_indices[c].pop()
            length_c = len(candidate_class_indices[c])
            if length_c == 0:
                del candidate_class_indices[c]
                candidate_anchor_class.remove(c)
            elif length_c < 2:
                candidate_anchor_class.remove(c)
            # select negative class
            candidate_negative_class = list(candidate_class_indices.keys())
            if length_c:
                candidate_negative_class.remove(c)
            # random.shuffle(candidate_negative_class)
            if len(candidate_negative_class) > self.negative_num:
                candidate_negative_class=random.sample(candidate_negative_class, self.negative_num)

            ns = []
            while len(ns) < self.negative_num and len(candidate_negative_class):
                candidate_negative_class_update = []
                for nc in candidate_negative_class:
                    n = candidate_class_indices[nc].pop()
                    ns.append(n)
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
