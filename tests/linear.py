import os
import sys
sys.path.append('..')
import argparse

import numpy as np

import chainer_sequential.binary.utils.binary_util as bu
from chainer_sequential.binary.links.link_binary_linear import BinaryLinear
from chainer_sequential.binary.links.link_batch_normalization import BatchNormalization

x = np.random.random((10,127)).astype(np.float32)-0.5
x = bu.binarize(x).astype(np.float32)
bl = BinaryLinear(127, 2)
bl.b.data = np.array([0.004, 0.006], dtype=np.float32)
bn = BatchNormalization(2)
bn.beta.data = np.array([-0.06, 0.01], dtype=np.float32)
bn.gamma.data = np.array([1.02, 1.2], dtype=np.float32)
bn(bl(x)) #init mean and var
res = bn(bl(x), test=True)
W = bl.W.data
print bu.np_to_uint8C(bu.binarize_real(x), 'A_in', 'row_major')
print bu.np_to_uint8C(bu.binarize_real(W.reshape(2, -1)), 'F_in', 'row_major', pad='1')
print bu.np_to_packed_uint8C(bu.binarize_real(res.data.flatten()), 'C_actual', 'row_major', pad='0')

print bu.np_to_floatC(bl.b.data.astype(np.float16), 'Bias', 'row_major')
print bu.np_to_floatC(bn.beta.data.astype(np.float16), 'Beta', 'row_major')
print bu.np_to_floatC(bn.gamma.data.astype(np.float16), 'Gamma', 'row_major')
print bu.np_to_floatC(bn.avg_mean.astype(np.float16), 'Mean', 'row_major')
print bu.np_to_floatC(np.sqrt(bn.avg_var).astype(np.float16), 'Std', 'row_major')
