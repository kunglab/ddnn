import os
import sys
sys.path.append('..')
import argparse

import numpy as np

import chainer_sequential.binary.utils.binary_util as bu


x = np.random.random((3, 7)).astype(np.float32)-0.5
print bu.np_to_packed_uint8C(bu.binarize_real(x), 'A_in', 'row_major')
print bu.np_to_packed_uint8C(bu.binarize_real(x[:1, 3:6]), 'C_actual', 'row_major')

print "===="



x = np.random.random((2, 2, 5, 5)).astype(np.float32)-0.5
print bu.np_to_packed_uint8C(bu.binarize_real(x), 'A_in', 'row_major')
print bu.np_to_packed_uint8C(bu.binarize_real(x[1,1,1:4,1:4]), 'C_actual', 'row_major')

print "===="


x = np.random.random((2, 2, 5, 5)).astype(np.float32)-0.5
x = bu.binarize(x)
y = np.random.random((2, 2, 3, 3)).astype(np.float32)-0.5
y = bu.binarize(y)
print bu.np_to_packed_uint8C(bu.binarize_real(x), 'A_in', 'row_major')
print bu.np_to_uint8C(bu.binarize_real(y.reshape(4, -1)), 'B_in', 'row_major', pad='1')
print "res: ", np.dot(x[0, :, 4:, 1:4].flatten(), y[0, :, 2, :].flatten())
