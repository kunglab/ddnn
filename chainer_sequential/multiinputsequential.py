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
    def __init__(self, ninputs, merge_function='max_pool_concat', stages=[0], weight_initializer="Normal", weight_init_std=1):
        self.ninputs = ninputs
        self.merge_function = merge_function
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

    def max_pool(self, hs):
        num_output = len(hs[0]) 
        houts = []
        for i in range(num_output):
            shape = hs[0][i].shape
            h = F.dstack([F.reshape(h[i],(shape[0], -1)) for h in hs])
            x = 1.0*F.max(h,2)
            x = F.reshape(x, shape)
            houts.append(x)
        return houts
    
    def max_pool_concat(self, hs):
        num_output = len(hs[0]) 
        houts = []
        i = 0
        x = F.max(F.dstack([h[i] for h in hs]),2)
        houts.append(x)
        for i in range(1,num_output):
            #x = 0
            #for h in hs:
            #    x = x + h[i]
            x = F.concat([h[i] for h in hs],1)
            houts.append(x) # Merged branch exit and main exit
        return houts
    
    def avg_pool(self, hs):
        num_output = len(hs[0]) 
        houts = []
        for i in range(num_output):
            shape = hs[0][i].shape
            h = F.dstack([F.reshape(h[i],(shape[0], -1)) for h in hs])
            x = 1.0*F.sum(h,2)/h.shape[2]
            x = F.reshape(x, shape)
            houts.append(x)
        return houts
    
    def avg_pool_concat(self, hs):
        num_output = len(hs[0]) 
        houts = []
        i = 0
        h = F.dstack([h[i] for h in hs])
        x = 1.0*F.sum(h,2)/h.shape[2]
        houts.append(x)
        for i in range(1,num_output):
            #x = 0
            #for h in hs:
            #    x = x + h[i]
            x = F.concat([h[i] for h in hs],1)
            houts.append(x) # Merged branch exit and main exit
        return houts
    
    def concat(self, hs):
        num_output = len(hs[0]) 
        houts = []
        for i in range(num_output):
            #x = 0
            #for h in hs:
            #    x = x + h[i]
            x = F.concat([h[i] for h in hs],1)
            houts.append(x) # Merged branch exit and main exit
        return houts
    
    def __call__(self, *inputs, **kwargs):
        test=kwargs.get('test',False)
        
        hs = [None]*len(self.inputs)
        for i,inp in enumerate(self.inputs):
            hs[i] = inp(inputs[i],test=test) # return a tuple of branch exit and main exit
        # 5 choices: max_pool, avg_pool, concat, max_pool_concat, avg_pool_concat
        # max_pool_concat is first output is max_pool later output(s) are concat
        # avg_pool_concat is first output is avg_pool later outpu(s) are concat
        houts = getattr(self,self.merge_function)(hs)
            
        # get result of the main exit
        tail = super(MultiInputSequential, self).__call__(houts[1], **kwargs)
        mainh = list(tail)
        
        # branch exit
        branchh = [h[0] for h in hs]+[houts[0]]
        return tuple(branchh+mainh)

    def generate_call(self):
        raise NotImplementedError()

    def generate_c(self, shape, name="main"):
        raise NotImplementedError()
