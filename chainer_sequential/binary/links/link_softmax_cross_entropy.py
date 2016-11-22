from __future__ import absolute_import

from chainer import link
from chainer.functions import softmax_cross_entropy

class SoftmaxCrossEntropy(link.Link):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.cname = "l_softmax_cross_entropy"

    def __call__(self, x, t):
        """Applies the softmax cross entropy.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the softmax cross entropy.

        """
        loss = softmax_cross_entropy(x,t)
        return loss
