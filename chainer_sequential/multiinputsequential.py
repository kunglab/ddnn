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
    # merge_function: one of max_pool, avg_pool, concat, max_pool_concat, avg_pool_concat
    def __init__(self, ninputs, merge_function='max_pool_concat', stages=[0], weight_initializer="Normal", weight_init_std=1):
        self.ninputs = ninputs
        self.merge_function = merge_function
        self.inputs = []
        self.local = None
        super(MultiInputSequential, self).__init__(stages, weight_initializer, weight_init_std)
    
    def add_input(self, inp):
        self.inputs.append(inp)
    def add_local(self, local):
        self.local = local  
        
    def to_dict(self):
        layers = []
        inputs = []
        for inp in self.inputs:
            inputs.append(inp.to_dict())
        
        result = super(MultiInputSequential, self).to_dict()
        result["inputs"] = inputs
        if self.local is not None:
            result["local"] = self.local.to_dict()
        return result

    def from_json(self, str):
        self.inputs = []
        self.local = None
        super(MultiInputSequential, self).from_json(str)
        
    def from_dict(self, dict):
        if dict.get('local'):
            layer = Sequential()
            layer.from_dict(dict['local'])
            self.local = layer
        for i, input_dict in enumerate(dict["inputs"]):
            layer = Sequential()
            layer.from_dict(input_dict)
            self.inputs.append(layer)
        super(MultiInputSequential, self).from_dict(dict)
    
    def avg_pool_max_pool(self, hs):
        num_output = len(hs[0]) 
        houts = []
        i = 0
        shape = hs[0][i].shape
        h = F.dstack([F.reshape(h[i],(shape[0], -1)) for h in hs])
        x = 1.0*F.sum(h,2)/h.shape[2]
        x = F.reshape(x, shape)
        houts.append(x)
        
        for i in range(1,num_output):
            shape = hs[0][i].shape
            h = F.dstack([F.reshape(h[i],(shape[0], -1)) for h in hs])
            x = 1.0*F.max(h,2)
            x = F.reshape(x, shape)
            houts.append(x)
        return houts
    
    def max_pool_avg_pool(self, hs):
        num_output = len(hs[0]) 
        houts = []
        i = 0
        shape = hs[0][i].shape
        h = F.dstack([F.reshape(h[i],(shape[0], -1)) for h in hs])
        x = 1.0*F.max(h,2)
        x = F.reshape(x, shape)
        houts.append(x)
        
        for i in range(1,num_output):
            shape = hs[0][i].shape
            h = F.dstack([F.reshape(h[i],(shape[0], -1)) for h in hs])
            x = 1.0*F.sum(h,2)/h.shape[2]
            x = F.reshape(x, shape)
            houts.append(x)
        return houts
    
    def concat_max_pool(self, hs):
        num_output = len(hs[0]) 
        houts = []
        i = 0
        x = F.concat([h[i] for h in hs],1)
        houts.append(x)
        for i in range(1,num_output):
            shape = hs[0][i].shape
            h = F.dstack([F.reshape(h[i],(shape[0], -1)) for h in hs])
            x = 1.0*F.max(h,2)
            x = F.reshape(x, shape)
            houts.append(x)
        return houts
    
    def concat_avg_pool(self, hs):
        num_output = len(hs[0]) 
        houts = []
        i = 0
        x = F.concat([h[i] for h in hs],1)
        houts.append(x)
        for i in range(1,num_output):
            shape = hs[0][i].shape
            h = F.dstack([F.reshape(h[i],(shape[0], -1)) for h in hs])
            x = 1.0*F.sum(h,2)/h.shape[2]
            x = F.reshape(x, shape)
            houts.append(x)
        return houts
        
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
    
    # in the number of values transmitted (divide by 32 if it is in binary)
    def get_communication_costs(self):
        if hasattr(self,'exit_size'):
            return self.exit_size
        else:
            return [0,0]
    
    # in bits
    def get_device_memory_cost(self):
        sequence = self.inputs[0]
        cost = 0
        for i, link in enumerate(sequence.links):
            if isinstance(link, chainer.link.Link):
                for l in link.links():
                    if hasattr(l, 'W'):
                        cost += np.prod(l.W.data.shape)
                    elif hasattr(l, 'b'):
                        cost += 32*np.prod(l.b.data.shape)
                    elif hasattr(l, 'gamma'):
                        cost += 32
                    elif hasattr(l, 'beta'):
                        cost += 32
                    elif hasattr(l, 'avg_mean'):
                        cost += 32
                    elif hasattr(l, 'avg_var'):
                        cost += 32
            elif isinstance(link, Sequential):
                for j, link in enumerate(link.links):
                    if isinstance(link, chainer.link.Link):
                        for l in link.links():
                            if hasattr(l, 'W'):
                                cost += np.prod(l.W.data.shape)
                            elif hasattr(l, 'b'):
                                cost += 32*np.prod(l.b.data.shape)
                            elif hasattr(l, 'gamma'):
                                cost += 32
                            elif hasattr(l, 'beta'):
                                cost += 32
                            elif hasattr(l, 'avg_mean'):
                                cost += 32
                            elif hasattr(l, 'avg_var'):
                                cost += 32
                            #elif hasattr(l, 'N'):
                            #    cost += 8*b
        return cost
    
    def predict(self, *inputs, **kwargs):
        ent_Ts=kwargs.get('ent_Ts',None)
        test=kwargs.get('test',True)
        
        hs = [None]*len(self.inputs)
        for i,inp in enumerate(self.inputs):
            hs[i] = inp(inputs[i], test=test) # return a tuple of branch exit and main exit
            self.exit_size = [None]*len(hs[i])
            for j in range(len(self.exit_size)):
                self.exit_size[j] = np.prod(hs[i][j].shape[1:])
                
        houts = getattr(self,self.merge_function)(hs)
        if self.local is not None:
            r = self.local(houts[0], test=test)
            houts[0] = r[0]

        locally_exited = self.entropy_exit(houts[0], ent_Ts[0])
        
        exits = []
        ex = np.sum(locally_exited).tolist()
        exits.append(ex)
        
        ys, exited = self.predict_with_mask(houts[-1], ent_Ts=ent_Ts[min(1,len(ent_Ts)-1):], test=True)
        
        if hasattr(ys.data, 'get'):
            ysdata = ys.data.get()
        else:
            ysdata = ys.data
        for j,lex in enumerate(locally_exited):
            if lex:
                if hasattr(houts[0][j].data, 'get'):
                    h = houts[0][j].data.get()
                else:
                    h = houts[0][j].data
                ysdata[j] = h
        
        with cuda.get_device(ys.data):
            ys.data = cuda.cupy.asarray(ysdata)
        
        exited = np.array(exited)
        exited = exited[locally_exited==False]
        ex = np.sum(exited).tolist()
        total = len(exited)
        exits.append(ex)
        exits.append(total-ex)
        return ys, exits
    
    def __call__(self, *inputs, **kwargs):
        test=kwargs.get('test',False)
        
        hs = [None]*len(self.inputs)
        for i,inp in enumerate(self.inputs):
            hs[i] = inp(inputs[i], test=test) # return a tuple of branch exit and main exit
            self.exit_size = [None]*len(hs[i])
            for j in range(len(self.exit_size)):
                self.exit_size[j] = np.prod(hs[i][j].shape[1:])
        
        # 5 choices: max_pool, avg_pool, concat, max_pool_concat, avg_pool_concat
        # max_pool_concat is first output is max_pool later output(s) are concat
        # avg_pool_concat is first output is avg_pool later outpu(s) are concat
        houts = getattr(self,self.merge_function)(hs)
        if self.local is not None:
            r = self.local(houts[0], test=test)
            houts[0] = r[0]
            
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
