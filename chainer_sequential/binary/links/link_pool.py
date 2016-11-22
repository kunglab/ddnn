from __future__ import absolute_import
from chainer import link

import chainer.functions as F

class Pool2D(link.Link):
    def __init__(self, kern=3, stride=2, pad=0):
        self.kern = kern
        self.stride = stride
        self.pad = pad
        super(Pool2D, self).__init__()
        self.cname = "l_pool"

    def __call__(self, x):
        return F.max_pooling_2d(x, self.kern, stride=self.stride, pad=self.pad, cover_all=False)
