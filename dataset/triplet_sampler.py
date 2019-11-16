# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        super(RandomIdentitySampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        # self.index_dic为字典，每一个键值对应一个list，list中存放属于该类别的数据索引
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        # self.pids: 所有的类别
        self.pids = list(self.index_dic.keys())

        '''
        estimate number of examples in an epoch；对于类别中样本总数不足num_instances的，按照num_instances计数；对于类别数大于等于
        num_instances的，按照num - num % self.num_instances计数，也就是int(num/self.num_instances) * self.num_instances
        '''
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            # 如果某个类别的样本总数不足self.num_instances
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        # 对于每一类数据
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            # 如果某个类别的样本总数不足self.num_instances，则采用重采样的方式；replace=True可以反复选取同一个元素
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            # 随机打乱idxs，因为idxs是深copy过来的，所以self.index_dic不受影响
            random.shuffle(idxs)
            batch_idxs = []

            # 假设pid类中所有数据可以划分n个num_instances(多余的舍弃)，batch_idxs_dict[pid]为长度为n的列表，类标中的每一个值为num_instances个下标
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        # final_idxs为一个list，每batch size个数据均由 num_instances * self.num_pids_per_batch组成
        # 下面这个判断语句会导致有几类看不到，并且导致len(final_idxs)和上面的self.length不一致
        while len(avai_pids) >= self.num_pids_per_batch:
            # 从所有类别中选中self.num_pids_per_batch
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                # 当该类数据已经被取完的时候，则将该类pop出去
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        # self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        super(RandomIdentitySampler_alignedreid, self).__init__(data_source)
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances