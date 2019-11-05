# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    # torch.norm：axis参数缩减的维度
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    欧式距离计算公式：对于两个n维向量，
    math:: d(x,y) = \sqrt {(x_1-y_1)^2 + (x_2-y_2)^2  + ... + (x_n-y_n)^2} = \sqrt {\sum_1^n (x_i - y_i)^2}

    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    # m, n代表第一个矩阵和第二个矩阵各有多少个样本
    m, n = x.size(0), y.size(0)
    # torch.pow 对张量按照元素求幂；sum函数中 axis参数缩减的维度；expand：将数据维度从[m, 1]每一个元素复制 n 遍组成一行，最终扩充到[m, n]
    # 这句话的作用为求出每一个d维向量的平方和，维度为[m, 1]，然后扩充到[m, n]
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # 这句话的作用为求出每一个d维向量的平方和，维度为[n, 1]，然后扩充到[n, m]，然后转置到[m, n]
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    # + 表示按元素相加
    dist = xx + yy
    # addmm_ 是Inplace的操作，在原对象基本上进行更改；这句话的作用为将dist中的每一个元素 乘以1，然后减去2倍的 x与y.t() 的矩阵乘法得到的
    # 对应位置的值，公式为 math:: 1*dist - 2*(x.mm(y.t()))，得到结果中的每一值为 \sum_1^n (x_i - y_i)^2
    dist.addmm_(1, -2, x, y.t())
    # clamp夹紧，当值小于1e-12的时候置为1e-12，sqrt 按元素求 开方
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    # dist_mat为对称矩阵，表示N个样本两两之间的距离
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    '''
    labels的维度为[N]，expand操作，将其每一个值复制N遍组成一列，最终扩充为[N, N]
    is_pos表示样本两两之间是否相似；is_neg表示样本两两之间是否不相似；两者的维度均为[N, N]；类型均为torch.bool
    labels.expand(N, N)以及其对称矩阵
    '''
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    '''
    `dist_ap` means distance(anchor, positive)，both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_mat[is_pos]得到的维度为[sum(is_pos)]，is_pos为True则取dist_mat相同位置的值。max(axis=1)表示取每一行的最大值
    值得注意的是，在原作者代码中重写了sampler，保证了sum(is_pos)%N=0
    '''
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    '''
    `dist_an` means distance(anchor, negative)
    both `dist_an` and `relative_n_inds` with shape [N, 1]
    '''
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        """

        :param global_feat: batch个样本的feature；类型为tensor；维度为[batch_size, feature_dim]
        :param labels: ground truth labels with shape (num_classes)
        :param normalize_feature: 是否对每个样本的feature做正则归一化；类型为bool
        :return loss: 计算出的loss值
        :return dist_ap第i个值：在与第i个样本类别相同的样本中取出距离最远的距离值；类型为tensor；维度为[batch_size]
        :return dist_an第i个值：在与第i个样本类别不同的样本中取出距离最近的距离值；类型为tensor；维度为[batch_size]
        """
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: q_i = (1 - epsilon) * a_i + epsilon / N.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        '''
        scatter_第一个参数为1表示分别对每行填充；targets.unsqueeze(1)得到的维度为[num_classes, 1]；
        填充方法为：取出targets的第i行中的第一个元素（每行只有一个元素），记该值为j；则前面tensor中的(i,j)元素填充1；
        最终targets的维度为[batch_size, num_classes]，每一行代表一个样本，若该样本类别为j，则只有第j元素为1，其余元素为0
        '''
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # mean(0)表示缩减第0维，也就是按列求均值，最终维度为[num_classes]，得到该batch内每一个类别的损失，再求和
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLossOrigin(nn.Module):
    """
    Reference: https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py
    """
    def __init__(self, margin=0):
        super(TripletLossOrigin, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def __call__(self, global_feat, labels, normalize_feature=False):
        """

        :param global_feat: batch个样本的feature；类型为tensor；维度为[batch_size, feature_dim]
        :param labels: ground truth labels with shape (num_classes)
        :param normalize_feature: 是否对每个样本的feature做正则归一化；类型为bool
        :return loss: 计算出的loss值
        :return prec：一个batch内的准确率；对于第i个样本，若与之类别不同样本中最近距离 大于 与之类别相同样本中的最远距离，则判断正确。
        """
        if len(torch.unique(labels)) == 1:  # TODO
            return 0, 0
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist = euclidean_dist(global_feat, global_feat)
        n = global_feat.size(0)

        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # new() 创建一个same data type的tensor，维度为[0]；resize_as_将前面的tensor维度转为和dist_an维度相同（注意不是reshape）；fill_填充1
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        '''
        Compute ranking hinge loss；dist_an、dist_ap和y的维度均为[N]；Pytorch中的接口函数：loss(x,y) = max(0, −y∗(x1−x2)+margin)
        dist_ap第i个值：在与第i个样本类别相同的样本中取出距离最远的距离值；dist_an第i个值：在与第i个样本类别不同的样本中取出距离最近的距离值

        math:: loss(x,y) = max(d(a,p)-d(a,n)+margin, 0) 其中，a表示anchor的feature；p表示positive样本的feature；n表示negative样本的feature
        '''
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # 对于第i个样本，若与之类别不同样本中最近距离 大于 与之类别相同样本中的最远距离，则判断正确。
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
