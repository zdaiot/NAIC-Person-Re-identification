from __future__ import print_function

import sys
import os.path as osp
import timeit
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + '/../')
from eval_reid import eval_func
"""
Test the speed of cython-based evaluation code. The speed improvements
can be much bigger when using the real reid data, which contains a larger
amount of query and gallery images.

Note: you might encounter the following error:
  'AssertionError: Error: all query identities do not appear in gallery'.
This is normal because the inputs are random numbers. Just try again.
"""

print('*** Compare running time ***')

setup = '''
import numpy as np
from eval_reid import eval_func
num_q = 30
num_g = 300
max_rank = 5
num_return = 200
distmat = np.random.rand(num_q, num_g) * 20
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
'''

pytime = timeit.timeit('eval_func(distmat, q_pids, g_pids, max_rank, num_return, use_cython=False)', setup=setup, number=20)
cytime = timeit.timeit('eval_func(distmat, q_pids, g_pids, max_rank, num_return, use_cython=True)', setup=setup, number=20)
print('Python time: {} s'.format(pytime))
print('Cython time: {} s'.format(cytime))
print('Cython is {} times faster than python\n'.format(pytime / cytime))


print("=> Check precision")

num_q = 30
num_g = 300
max_rank = 5
num_return = 200
distmat = np.random.rand(num_q, num_g) * 20
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)

cmc, mAP, allAP = eval_func(distmat, q_pids, g_pids, max_rank, num_return, use_cython=False)
print("Python:\nmAP = {} \ncmc = {} \nallAP = {} \n".format(mAP, cmc, allAP))
cmc, mAP, allAP = eval_func(distmat, q_pids, g_pids, max_rank, num_return, use_cython=True)
print("Cython:\nmAP = {} \ncmc = {} \nallAP = {} \n".format(mAP, cmc, allAP))
