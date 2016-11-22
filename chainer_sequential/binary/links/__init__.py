from chainer import link

class CLink(object):
    def generate_c(self):
        raise NotImplementedError("Not implemented. This link cannot be exported as c.")
       
    def param_mem(self):
        raise NotImplementedError("Not implemented. This link cannot be exported as c.")
    
    def temp_mem(self):
        raise NotImplementedError("Not implemented. This link cannot be exported as c.")
    
from link_bst import BST
from link_pool import Pool2D
from link_batch_normalization import BatchNormalization
from link_binary_convolution import BinaryConvolution2D
from link_binary_linear import BinaryLinear
from link_softmax_cross_entropy import SoftmaxCrossEntropy

from link_linear_BN_BST import LinearBNBST
from link_binary_linear_BN_BST import BinaryLinearBNBST

from link_binary_linear_softmax_layer import BinaryLinearSoftmax
from link_binary_linear_BN_softmax_layer import BinaryLinearBNSoftmax

from link_conv_BN_BST import ConvBNBST
from link_binary_conv_BN_BST import BinaryConvBNBST

from link_conv_pool_BN_BST import ConvPoolBNBST
from link_binary_conv_pool_BN_BST import BinaryConvPoolBNBST

