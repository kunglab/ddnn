import numpy
import chainer
import weightnorm
import util

class Link(object):
    def __call__(self, x):
        raise NotImplementedError()

    def has_multiple_weights(self):
        return False

    def from_dict(self, dict):
        for attr, value in dict.iteritems():
            setattr(self, attr, value)

    def to_dict(self):
        dict = {}
        for attr, value in self.__dict__.iteritems():
            dict[attr] = value
        return dict

    def to_chainer_args(self):
        dict = {}
        for attr, value in self.__dict__.iteritems():
            if attr[0] == "_":
                pass
            else:
                dict[attr] = value
        return dict

    def to_link(self):
        raise NotImplementedError()

    def dump(self):
        print "Link: {}".format(self._link)
        for attr, value in self.__dict__.iteritems():
            print "    {}: {}".format(attr, value)

class Convolution2D(Link):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, bias=0, nobias=False, use_cudnn=True, use_weightnorm=False):
        self._link = "Convolution2D"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.bias = bias
        self.nobias = nobias
        self.use_cudnn = use_cudnn
        self.use_weightnorm = use_weightnorm

    def to_link(self):
        args = self.to_chainer_args()
        del args["use_weightnorm"]
        if self.use_weightnorm:
            if hasattr(self, "_initialW"):
                args["initialV"] = self._initialW
            return weightnorm.Convolution2D(**args)

        if hasattr(self, "_initialW"):
            args["initialW"] = self._initialW
        return chainer.links.Convolution2D(**args)

class Deconvolution2D(Link):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, bias=0, nobias=False, outsize=None, use_cudnn=True, use_weightnorm=False):
        self._link = "Deconvolution2D"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.bias = bias
        self.nobias = nobias
        self.outsize = outsize
        self.use_cudnn = use_cudnn
        self.use_weightnorm = use_weightnorm

    def to_link(self):
        args = self.to_chainer_args()
        del args["use_weightnorm"]
        if self.use_weightnorm:
            if hasattr(self, "_initialW"):
                args["initialV"] = self._initialW
            return weightnorm.Deconvolution2D(**args)
        if hasattr(self, "_initialW"):
            args["initialW"] = self._initialW
        return chainer.links.Deconvolution2D(**args)

class DilatedConvolution2D(Link):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, dilate=1, bias=0, nobias=False, use_cudnn=True):
        self._link = "DilatedConvolution2D"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.dilate = dilate
        self.bias = bias
        self.nobias = nobias
        self.use_cudnn = use_cudnn

    def to_link(self):
        args = self.to_chainer_args()
        if hasattr(self, "_initialW"):
            args["initialW"] = self._initialW
        return chainer.links.DilatedConvolution2D(**args)

class EmbedID(Link):
    def __init__(self, in_size, out_size, ignore_label=None):
        self._link = "EmbedID"
        self.in_size = in_size
        self.out_size = out_size
        self.ignore_label = ignore_label

    def to_link(self):
        args = self.to_chainer_args()
        if hasattr(self, "_initialW"):
            args["initialW"] = self._initialW
        return chainer.links.EmbedID(**args)

class GRU(Link):
    def __init__(self, n_units, n_inputs=None):
        self._link = "GRU"
        self.n_units = n_units
        self.n_inputs = n_inputs

    def to_link(self):
        args = self.to_chainer_args()
        if hasattr(self, "_init"):
            args["init"] = self._init
        if hasattr(self, "_inner_init"):
            args["inner_init"] = self._inner_init
        return chainer.links.GRU(**args)

    def has_multiple_weights(self):
        return True

class Linear(Link):
    def __init__(self, in_size, out_size, bias=0, nobias=False, use_weightnorm=False):
        self._link = "Linear"
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.nobias = nobias
        self.use_weightnorm = use_weightnorm

    def to_link(self):
        args = self.to_chainer_args()
        del args["use_weightnorm"]
        if self.use_weightnorm:
            if hasattr(self, "_initialW"):
                args["initialV"] = self._initialW
            return weightnorm.Linear(**args)
        if hasattr(self, "_initialW"):
            args["initialW"] = self._initialW
        return chainer.links.Linear(**args)

class LSTM(Link):
    def __init__(self, in_size, out_size):
        self._link = "LSTM"
        self.in_size = in_size
        self.out_size = out_size

    def to_link(self):
        args = self.to_chainer_args()
        if hasattr(self, "_lateral_init"):
            args["lateral_init"] = self._lateral_init
        if hasattr(self, "_upward_init"):
            args["upward_init"] = self._upward_init
        if hasattr(self, "_bias_init"):
            args["bias_init"] = self._bias_init
        if hasattr(self, "_forget_bias_init"):
            args["forget_bias_init"] = self._forget_bias_init
        return chainer.links.GRU(**args)

    def has_multiple_weights(self):
        return True

class StatelessLSTM(Link):
    def __init__(self, in_size, out_size):
        self._link = "StatelessLSTM"
        self.in_size = in_size
        self.out_size = out_size

    def to_link(self):
        args = self.to_chainer_args()
        if hasattr(self, "_lateral_init"):
            args["lateral_init"] = self._lateral_init
        if hasattr(self, "_upward_init"):
            args["upward_init"] = self._upward_init
        return chainer.links.GRU(**args)

    def has_multiple_weights(self):
        return True

class StatefulGRU(Link):
    def __init__(self, in_size, out_size, bias_init=0):
        self._link = "StatefulGRU"
        self.in_size = in_size
        self.out_size = out_size
        self.bias_init = bias_init

    def to_link(self):
        args = self.to_chainer_args()
        if hasattr(self, "_init"):
            args["init"] = self._init
        if hasattr(self, "_inner_init"):
            args["inner_init"] = self._inner_init
        return chainer.links.GRU(**args)

    def has_multiple_weights(self):
        return True

class StatefulPeepholeLSTM(Link):
    def __init__(self, in_size, out_size):
        self._link = "StatefulPeepholeLSTM"
        self.in_size = in_size
        self.out_size = out_size

    def to_link(self):
        args = self.to_chainer_args()
        return chainer.links.StatefulPeepholeLSTM(**args)

class BatchNormalization(Link):
    def __init__(self, size, decay=0.9, eps=2e-05, dtype="float32", use_gamma=True, use_beta=True, use_cudnn=True):
        self._link = "BatchNormalization"
        self.size = size
        self.decay = decay
        self.eps = eps
        self.dtype = dtype
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.use_cudnn = use_cudnn

    def to_link(self):
        args = self.to_chainer_args()
        if args["dtype"] == "float32":
            args["dtype"] = numpy.float32
        elif args["dtype"] == "float64":
            args["dtype"] = numpy.float64
        elif args["dtype"] == "float16":
            args["dtype"] = numpy.float16
        return chainer.links.BatchNormalization(**args)
