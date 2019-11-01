# encoding: utf-8

import numpy as np
from evaluate.eval_cylib.eval_metrics_cy import evaluate_cy


def eval_func(distmat, q_pids, g_pids, max_rank=50, num_return=200, use_cython=False):
    """ Evaluation

    :param distmat: 查询集与数据库之间的距离矩阵；类型为numpy；维度为[num_q, num_g]
    :param q_pids: 查询集的类标；类型为numpy；维度为[num_q]
    :param g_pids: 数据库的类标；类型为numpy；维度为[num_g]
    :param max_rank: 计算1～max_rank的准确率；类型为int
    :param num_return: 返回查询到的前 num_return 个结果；类型为int
    :param use_cython: 是否使用C语言版本的评价指标代码；类型为bool
    :return all_rank_precison: 1～max_rank的准确率；类型为numpy，维度为[max_rank, ]
    :return mAP: 平均检索精度；类型为float
    :return all_AP: 每一个查询样本的检索精度；类型为list；维度为[num_q]
    """
    if use_cython:
        print('USE CPYTHON!')
        return evaluate_cy(distmat, q_pids, g_pids, max_rank, num_return)
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    assert num_g >= num_return, "数据库中的图片总数不足num_return！"
    assert num_return >= max_rank, "num_return必须大于等于max_rank"
    # 返回数据中的每一行按行从小到大排序后的位置索引
    indices = np.argsort(distmat, axis=1)
    '''
    取indices中的每一行，将数据库类标重排，排序后为二维矩阵，矩阵大小为[num_q, num_g]，第i行代表所有数据库中样本按与第i个查询样本距离远近
    重排后的类标；然后第i行所有元素与第i个查询集的真实类标进行对比；最终得到matches的第i行表示所有的数据库按照与第i个查询数据距离远近重排后，
    每个数据库样本的类标是否真的与第i个查询集的类标相同
    '''
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    matches = matches[:, :num_return]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    # 对于每个查询样本
    for q_idx in range(num_q):
        # 得到该查询样本与数据库的查询比对结果
        q_idx_match = matches[q_idx]

        # 分别求前i个数据的和，例如array([3, 5, 1])，得到array([3, 8, 9])。所以这里得到了返回的前i个结果中有多少个正确的，维度为[num_return]
        cmc = q_idx_match.cumsum()
        # 对cmc中的大于1的元素置为1，其余不变；cmc第i个元素表示前i个元素是否检索到了正确样本
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_ranking = q_idx_match.sum()
        tmp_cmc = q_idx_match.cumsum()
        '''
        查准率计算公式为 对于 每一个检索到的相似图像在检索到的全部图片中的下标 查准率=检索到的相似图片个数/当前检索到的图片总数
        首先先计算每一个查准率；然后与q_idx_match按元素相乘，若第i个结果为正确的，则计算查准率，否则不计算查准率
        '''
        precison_all = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        precison_ranking = np.asarray(precison_all) * q_idx_match
        # 计算AP，对于每一个ranking下的查准率求和，然后除以总的ranking数目
        if num_ranking == 0:
            AP = 0
        else:
            AP = precison_ranking.sum() / num_ranking
        all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    # 对all_cmc按列求和，最终得到的维度为[max_rank]；除以num_q后第一个元素表示rank1的正确率，依次类推
    all_rank_precison = all_cmc.sum(0) / num_q
    # 计算平均检索精度
    mAP = np.mean(all_AP)

    return all_rank_precison, mAP, all_AP
