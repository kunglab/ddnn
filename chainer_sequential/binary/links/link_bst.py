from __future__ import absolute_import

from chainer import link

from ..functions import function_bst

class BST(link.Link):
    def __init__(self):
        super(BST, self).__init__()
        self.cname = "l_bst"

    def __call__(self, x):
        """Applies the binary activation.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the binary activation.

        """
        return function_bst.bst(x)
