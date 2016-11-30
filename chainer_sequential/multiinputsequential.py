import copy, json, types
import chainer
import link
import function
import binary_link
import binary_function
import numpy as np
from chainer import cuda
import inspect
from chainer_ext.functions import entropy
import chainer.functions as F
from chainer import Variable
from .sequential import Sequential
                
class MultiInputSequential(Sequential):
    def __init__(self, ninputs, mergeFunction=F.average_pooling_2d, stages=[0], weight_initializer="Normal", weight_init_std=1):
        self.ninputs = ninputs
        self.mergeFunction = mergeFunction
        self.inputs = []
        super(MultiInputSequential, self).__init__(stages, weight_initializer, weight_init_std)
    
    def add_input(self, inp):
        self.inputs.append(inp)
        
    def to_dict(self):
        layers = []
        inputs = []
        for inp in self.inputs:
            inputs.append(inp.to_dict())
        
        result = super(MultiInputSequential, self).to_dict()
        result["inputs"] = inputs
        return result

    def from_json(self, str):
        self.inputs = []
        super(MultiInputSequential, self).from_json(str)
        
    def from_dict(self, dict):
        for i, input_dict in enumerate(dict["inputs"]):
            layer = Sequential()
            layer.from_dict(input_dict)
            self.inputs.append(layer)
        super(MultiInputSequential, self).from_dict(dict)

    def predict(self, *inputs, **kwargs):
        ent_T=kwargs.get('ent_T',None)
        ent_Ts=kwargs.get('ent_Ts',None)
        test=kwargs.get('test',True)
        
        # Return last layer result
        if ent_T is None and ent_Ts is None:
            return self(x, test)
        elif ent_Ts is None:
            ent_Ts = [ent_T]
        
        # TODO. Current pass all to the main branch
        return super(MultiInputSequential, self).__call__(inputs[-1], ent_T=None, ent_Ts=None, test=True)

    def __call__(self, *inputs, **kwargs):
        test=kwargs.get('test',False)
        
        hs = []
        for i,inp in enumerate(self.inputs):
            hs[i] = inp(inputs[i],test=test) # return a tuple of branch exit and main exit
        
        num_output = len(hs[0])
        
        # TODO: support other merge functions
        houts = []
        for i in range(num_output):
            x = 0
            for h in hs:
                x = x + h[i]
            houts.append(x) # Merged branch exit and main exit
                    
        # get result of the main exit
        mainh = list(super(MultiInputSequential, self).__call__(houts[1]))
        
        # branch exit
        branchh = houts[0]
        return tuple([branchh]+mainh)

    def generate_call(self):
        raise NotImplementedError()

    def generate_c(self, shape, name="main"):
        raise NotImplementedError()
