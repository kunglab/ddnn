from __future__ import absolute_import
import math

import numpy as np
import chainer

from ..links import CLink
from ..links import BinaryConvolution2D
from ..links import Pool2D
from ..links import BatchNormalization
from ..links import BST
from ..utils import binary_util as bu

class BinaryConvPoolBNBST(chainer.Chain, CLink):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0, pksize=3, pstride=2, ppad=0):
        super(BinaryConvPoolBNBST, self).__init__(
            bconv=BinaryConvolution2D(in_channels, out_channels, ksize=ksize, stride=stride, pad=pad),
            pool=Pool2D(pksize,pstride,ppad),
            bn=BatchNormalization(out_channels),
            bst=BST()
        )
        self.cname = "l_b_conv_pool_bn_bst"
        self.pksize = pksize
        self.ppad = ppad

    def __call__(self, h, test=False):
        h = self.bconv(h)
        h = self.pool(h)
        h = self.bn(h, test)
        h = self.bst(h)
        return h

    def generate_c(self, link_idx, inp_shape):
        #if not hasattr(self,'inp_shape'):
        #    raise Exception("no input shape found")
        #    return ""
        name = self.cname + str(link_idx)
        text = []
        m = 1
        w, h = inp_shape[2:4]
        sw, sh = self.bconv.stride
        pw, ph = self.bconv.pad
        pl_w, pl_h = self.pksize, self.pksize
        pl_sw, pl_sh = self.pool.stride, self.pool.stride
        pl_pw, pl_ph = self.ppad, self.ppad

        # Bconv
        l = self.bconv
        lname = name + '_' + l.name
        for p in l.params():
            pname = p.name
            if pname == 'W':
                num_f, n, kw, kh =  p.data.shape
                bin_data = bu.binarize_real(p.data).reshape(p.data.shape[0]*p.data.shape[1], -1)
                text += [bu.np_to_uint8C(bin_data, lname+'_'+pname, 'row_major', pad='1')]
            elif pname == 'b':
                text += [bu.np_to_floatC(p.data, lname+'_'+pname, 'row_major')]

        # BatchNormalization bn
        l = self.bn
        lName = l.name
        lname=name+'_'+lName
        for p in l.params():
            pname=p.name
            if pname == 'gamma':
                text += [bu.np_to_floatC(p.data, lname+'_'+pname, 'row_major')]
            elif pname == 'beta':
                text += [bu.np_to_floatC(p.data, lname+'_'+pname, 'row_major')]
        for p in l._persistent:
            pname=p
            persistent = l.__dict__[p]
            if pname == 'avg_mean':
                text += [bu.np_to_floatC(persistent, lname+'_mean', 'row_major')]
            elif pname == 'avg_var':
                text += [bu.np_to_floatC(np.sqrt(persistent, dtype=persistent.dtype), lname+'_std', 'row_major')]

        text = "\n".join(text)+'\n'
        ftext = "void {name}(uint8_t* input, uint8_t* output){{\n"
        ftext += "  bconv_layer(input, {name}_bconv_W, output, {name}_bconv_b, {name}_bn_gamma, {name}_bn_beta, {name}_bn_mean, {name}_bn_std, {m}, {num_f}, {w}, {h}, {n}, {kw}, {kh}, {sw}, {sh}, {pw}, {ph}, {pl_w}, {pl_h}, {pl_sw}, {pl_sh}, {pl_pw}, {pl_ph});\n}}\n\n"
        ftext = ftext.format(name=name, m=m, n=n, w=w, h=h, num_f=num_f, kw=kw,
                             kh=kh, sw=sw, sh=sh, pw=pw, ph=ph, pl_w=pl_w,
                             pl_h=pl_h, pl_sw=pl_sw, pl_sh=pl_sh, pl_pw=pl_pw,
                             pl_ph=pl_ph)
        text += ftext


        return text

    def param_mem(self):
        mem = 0.
        l = self.bconv
        for p in self.bconv.params():
            if p.name == 'W':
                num_f, n, kw, kh =  p.data.shape
                #Filters
                mem += num_f*n*kh*math.ceil(kw/8.)
                #Bias + BN
                mem += 5*num_f*32

        return mem

    def temp_mem(self, inp_shape):
        #TODO: UPDATE
        m, n, w, h = inp_shape
        sw, sh = self.bconv.stride
        for p in self.bconv.params():
            if p.name == 'W':
                _, _, kw, kh =  p.data.shape
                break

        res_w = (w - kw + 8) / 8
        res_h = h - kh + 1

        return m*n*res_w*res_h
