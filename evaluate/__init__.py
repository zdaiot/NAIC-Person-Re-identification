import numpy as np
import torch
from evaluate.eval_reid import eval_func
from evaluate.re_ranking import re_ranking


def euclidean_dist(x, y):
    """ 计算欧式距离
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist.numpy()


def re_rank(q, g):
    qq_dist = euclidean_dist(q, q)
    gg_dist = euclidean_dist(g, g)
    qg_dist = euclidean_dist(q, g)
    distmat = re_ranking(qg_dist, qq_dist, gg_dist)
    return distmat


def cos_dist(query_features, gallery_features):
    """ 计算余弦距离，相似性范围从-1到1：-1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
    0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。

    :param query_features: 查询样本集的features；类型为tensor；维度为[num_query, feature_dim]
    :param gallery_features: 数据库的features；类型为tensor；维度为[num_gallery, feature_dim]
    :return 查询集与数据库两两样本之间的余弦距离
    """
    qnorm = torch.norm(query_features, p=2, dim=1, keepdim=True)
    query_features = query_features.div(qnorm.expand_as(query_features))

    gnorm = torch.norm(gallery_features, p=2, dim=1, keepdim=True)
    gallery_features = gallery_features.div(gnorm.expand_as(gallery_features))

    score = torch.mm(query_features, gallery_features.t())
    score = score.cpu().numpy()
    return 1 - score
