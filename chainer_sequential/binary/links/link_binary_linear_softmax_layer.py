from __future__ import absolute_import

import chainer
import chainer.functions as F
import numpy as np

from ..links import CLink
from ..links import BinaryLinear
from ..links import BatchNormalization
from ..links import SoftmaxCrossEntropy
from ..utils import binary_util as bu
import math

class BinaryLinearSoftmax(chainer.Chain, CLink):
    def __init__(self, in_channels, out_channels):
        super(BinaryLinearSoftmax, self).__init__(
            bl=BinaryLinear(in_channels, out_channels),
            sm=SoftmaxCrossEntropy()
        )
        self.cname = "l_b_linear_softmax"

    def __call__(self, h, t=None):
        h = self.bl(h)
        if t is not None:
            self.accuracy = F.accuracy(h,t)
            loss = self.sm(h,t)
            return loss
        return h

    def generate_c(self, link_idx, inp_shape):
        name = self.cname + str(link_idx)
        text = []

        # BinaryLinear bl
        l = self.bl
        lName = l.name
        lname=name+'_'+lName
        for p in l.params():
            pname=p.name
            if pname == 'W':
                text += [bu.np_to_uint8C(bu.binarize_real(p.data.T), lname+'_'+pname, 'col_major', pad='1')]
                num_classes = p.data.shape[0]
                fc_size = p.data.shape[1]
            elif pname == 'b':
                text += [bu.np_to_floatC(p.data, lname+'_'+pname, 'row_major')]

        text = "\n".join(text)
        m = 1
        n = fc_size
        k = num_classes

        ftext = "void {name}(uint8_t* input, uint8_t* output){{\n"
        ftext += "  blinear_layer(input, {name}_bl_W, output, {name}_bl_b, {m}, {n}, {k}); \n}}\n\n"
        ftext = ftext.format(name=name, m=m, n=n, k=k)
        text += ftext

        return text

    def param_mem(self):
        mem = 0.
        for p in self.bl.params():
            if p.name == 'W':
                w, h = p.data.shape
                mem +=  math.ceil(h/8.)*w
                #Bias + BN
                mem += 5*h*32

        return mem

    def temp_mem(self, inp_shape):
        m = inp_shape[0]
        w = np.prod(inp_shape[1:])
        res_w = math.ceil(w/8.)
        return m*res_w
