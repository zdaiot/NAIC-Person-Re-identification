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
    return eval_market1501_cy(distmat, q_pids, g_pids, max_rank)

cpdef eval_market1501_cy(float[:,:] distmat, long[:] q_pids, long[:] g_pids, long max_rank):
    
    cdef long num_q = distmat.shape[0]
    cdef long num_g = distmat.shape[1]

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))
    
    cdef:
        long[:,:] indices = np.argsort(distmat, axis=1)
        long[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int64)

        float[:,:] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
        float[:] all_AP = np.zeros(num_q, dtype=np.float32)
        float num_valid_q = 0. # number of valid query

        long q_idx, q_pid, g_idx
        long[:] order = np.zeros(num_g, dtype=np.int64)
        long keep

        float[:] raw_cmc = np.zeros(num_g, dtype=np.float32) # binary vector, positions with value 1 are correct matches
        float[:] cmc = np.zeros(num_g, dtype=np.float32)
        long rank_idx
        unsigned long meet_condition

        float num_rel
        float[:] tmp_cmc = np.zeros(num_g, dtype=np.float32)
        float tmp_cmc_sum
    
    for q_idx in range(num_q):
        for g_idx in range(num_g):
            raw_cmc[g_idx] = matches[q_idx][g_idx]

        # compute cmc
        function_cumsum(raw_cmc, cmc, num_g)
        for g_idx in range(num_g):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1

        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        function_cumsum(raw_cmc, tmp_cmc, num_g)
        num_rel = 0
        tmp_cmc_sum = 0
        for g_idx in range(num_g):
            tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
            num_rel += raw_cmc[g_idx]
        all_AP[q_idx] = tmp_cmc_sum / num_rel

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    # compute averaged cmc
    cdef float[:] avg_cmc = np.zeros(max_rank, dtype=np.float32)
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q
    
    cdef float mAP = 0
    for q_idx in range(num_q):
        if not math.isnan(all_AP[q_idx]):
            mAP += all_AP[q_idx]
    mAP /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), mAP, all_AP


# Compute the cumulative sum
cdef void function_cumsum(cython.numeric[:] src, cython.numeric[:] dst, long n):
    cdef long i
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]
