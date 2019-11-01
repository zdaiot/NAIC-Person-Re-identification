# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import cython
import math
import numpy as np
cimport numpy as np

"""
Compiler directives:
https://github.com/cython/cython/wiki/enhancements-compilerdirectives

Cython tutorial:
https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
"""

# Main interface
cpdef evaluate_cy(distmat, q_pids, g_pids, max_rank):
    distmat = np.asarray(distmat, dtype=np.float32)
    q_pids = np.asarray(q_pids, dtype=np.int64)
    g_pids = np.asarray(g_pids, dtype=np.int64)
    return eval_cy(distmat, q_pids, g_pids, max_rank)

cpdef eval_cy(float[:,:] distmat, long[:] q_pids, long[:] g_pids, long max_rank):
    """ Evaluation

    :param distmat: 查询集与数据库之间的距离矩阵；类型为numpy；维度为[num_q, num_g]
    :param q_pids: 查询集的类标；类型为numpy；维度为[num_q]
    :param g_pids: 数据库的类标；类型为numpy；维度为[num_g]
    :param max_rank: 计算1～max_rank的准确率
    :return all_rank_precison: 1～max_rank的准确率
    :return mAP: 平均检索精度
    :return all_AP: 每一个ranking的检索精度
    """
    
    cdef long num_q = distmat.shape[0]
    cdef long num_g = distmat.shape[1]

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
    
    cdef:
        # 返回数据中的每一行按行从小到大排序后的位置索引
        long[:,:] indices = np.argsort(distmat, axis=1)
        # 取indices中的每一行，将数据库类标重排，排序后为二维矩阵，矩阵大小为[num_q, num_g]，第i行代表所有数据库中样本按与第i个查询样本距离远近
        # 重排后的类标；然后第i行所有元素与第i个查询集的真实类标进行对比；最终得到matches的第i行表示所有的数据库按照与第i个查询数据距离远近重排后，
        # 每个数据库样本的类标是否真的与第i个查询集的类标相同
        long[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int64)

        float[:,:] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
        float[:] all_AP = np.zeros(num_q, dtype=np.float32)

        long q_idx, g_idx

        float[:] raw_cmc = np.zeros(num_g, dtype=np.float32) # binary vector, positions with value 1 are correct matches
        float[:] cmc = np.zeros(num_g, dtype=np.float32)
        long rank_idx

        float num_ranking
        float[:] tmp_cmc = np.zeros(num_g, dtype=np.float32)
        float precison_ranking_sum
    # 对于每一个查询样本
    for q_idx in range(num_q):
        # 对于每一个数据库样本，得到该查询样本与数据库的查询比对结果raw_cmc
        for g_idx in range(num_g):
            raw_cmc[g_idx] = matches[q_idx][g_idx]

        # compute cmc，cmc得到了返回的前i个结果中有多少个正确的，维度为[num_g]
        function_cumsum(raw_cmc, cmc, num_g)
        # 对cmc中的大于1的元素置为1，其余不变；cmc第i个元素表示前i个元素是否检索到了正确样本
        for g_idx in range(num_g):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1
        # 在all_cmc的第q_idx行放入cmc的前max_rank的结果，方便下面计算rank1等
        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        # tmp_cmc得到了返回的前i个结果中有多少个正确的，维度为[num_g]
        function_cumsum(raw_cmc, tmp_cmc, num_g)
        num_ranking = 0
        precison_ranking_sum = 0
        for g_idx in range(num_g):
            # 查准率计算公式为 对于 每一个检索到的相似图像在检索到的全部图片中的下标 查准率=检索到的相似图片个数/当前检索到的图片总数
            # 首先先计算该g_idx下标查准率；然后与raw_cmc的对应元素相乘，若第g_idx个查询结果为正确的，则计算查准率，否则不计算查准率
            # 然后加到precison_ranking_sum上，求得所有ranking的查准率之和
            precison_ranking_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
            num_ranking += raw_cmc[g_idx]
        # 所有ranking的查准率之和除以ranking的总数得到当前查询样本的AP
        if num_ranking != 0:
            all_AP[q_idx] = precison_ranking_sum / num_ranking

    # compute averaged cmc
    cdef float[:] all_rank_precison = np.zeros(max_rank, dtype=np.float32)
    # 对于每一个rank
    for rank_idx in range(max_rank):
        # 有多少个查询样本前rank_idx个结果检索到了正确结果，放到avg_cmc[rank_idx]中
        for q_idx in range(num_q):
            all_rank_precison[rank_idx] += all_cmc[q_idx, rank_idx]
        # 除以查询集总数即可得到 rank_idx 的正确率
        all_rank_precison[rank_idx] /= num_q
    
    cdef float mAP = 0
    for q_idx in range(num_q):
        mAP += all_AP[q_idx]
    mAP /= num_q

    return np.asarray(all_rank_precison).astype(np.float32), mAP, np.asarray(all_AP).astype(np.float32)


# Compute the cumulative sum，即实现功能np.cumsum():分别求前i个数据的和，例如array([3, 5, 1])，得到array([3, 8, 9])
cdef void function_cumsum(cython.numeric[:] src, cython.numeric[:] dst, long n):
    cdef long i
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]
