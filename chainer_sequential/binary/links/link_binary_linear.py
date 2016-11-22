from __future__ import absolute_import

import numpy
from chainer import link

from ..functions import function_binary_linear
import math
from chainer import initializers
from chainer import cuda

class BinaryLinear(link.Link):
    """Binary Linear layer (a.k.a. binary fully-connected layer).

    This is a link that wraps the :func:`~chainer.functions.linear` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as
    parameters.

    The weight matrix ``W`` is initialized with i.i.d. Gaussian samples, each
    of which has zero mean and deviation :math:`\\sqrt{1/\\text{in_size}}`. The
    bias vector ``b`` is of size ``out_size``. Each element is initialized with
    the ``bias`` value. If ``nobias`` argument is set to True, then this link
    does not hold a bias vector.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        bias (float): Initial bias value.
        nobias (bool): If True, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.

    .. seealso:: :func:`~chainer.functions.linear`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """
    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(BinaryLinear, self).__init__()
        self.cname = "l_b_linear"

        # For backward compatibility
        self.initialW = initialW
        self.wscale = wscale

        self.out_size = out_size
        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        self._W_initializer = initializers._get_initializer(
            initialW, math.sqrt(wscale))

        if in_size is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_size)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_size, initializer=bias_initializer)

#    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
#                 initialW=None, initial_bias=None):
#        super(BinaryLinear, self).__init__(W=(out_size, in_size))
#        self.cname = "l_b_linear"
#
#        if initialW is None:
#            initialW = numpy.random.normal(
#                0, wscale * numpy.sqrt(1. / in_size), (out_size, in_size))
#        self.W.data[...] = initialW
#
#        if nobias:
#            self.b = None
#        else:
#            self.add_param('b', out_size)
#            if initial_bias is None:
#                initial_bias = bias
#            self.b.data[...] = initial_bias

    def _initialize_params(self, in_size):
        self.add_param('W', (self.out_size, in_size),
                       initializer=self._W_initializer)
    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.size // len(x.data))
        return function_binary_linear.binary_linear(x, self.W, self.b)
