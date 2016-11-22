import math
from chainer import cuda, Variable
from chainer import functions as F
from .binary.functions import function_bst 

class Function(object):
	def __call__(self, x):
		raise NotImplementedError()

	def from_dict(self, dict):
		for attr, value in dict.iteritems():
			setattr(self, attr, value)

	def to_dict(self):
		dict = {}
		for attr, value in self.__dict__.iteritems():
			dict[attr] = value
		return dict

class BinaryActivation(object):
	def __init__(self, nonlinearity="bst"):
		self.nonlinearity = nonlinearity

	def to_function(self):
		if self.nonlinearity.lower() == "bst":
			return bst()
		raise NotImplementedError()

class bst(Function):
	def __init__(self):
		self._function = "bst"

	def __call__(self, x):
		return function_bst.bst(x)

