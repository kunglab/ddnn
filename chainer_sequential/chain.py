import os

import numpy as np
from scipy.stats import entropy
import chainer
import sequential
from chainer import optimizers, serializers, Variable
from chainer import functions as F
from chainer import reporter
from chainer.functions.activation.softmax import softmax
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer_ext import weight_clip

class Chain(chainer.Chain):

    def __init__(self, compute_accuracy=True, lossfun=softmax_cross_entropy.softmax_cross_entropy, branchweight=1, branchweights=None, ent_T=0.1, ent_Ts=None,
                 accfun=accuracy.accuracy):
        super(Chain,self).__init__()
        #branchweights = [1000]*7+[1000]
        self.lossfun = lossfun
        if branchweight is not None and branchweights is None:
            self.branchweights = [branchweight]
        else:
            self.branchweights = branchweights
        if ent_T is not None and ent_Ts is None:
            self.ent_Ts = [ent_T]
        else:
            self.ent_Ts = ent_Ts
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.compute_accuracy = compute_accuracy

    def add_sequence(self, sequence):
        if isinstance(sequence, sequential.Sequential) == False:
            raise Exception()
        for i, link in enumerate(sequence.links):
            if isinstance(link, chainer.link.Link):
                self.add_link("link_{}".format(i), link)
                #print(link.name, link)
            elif isinstance(link, sequential.Sequential):
                for j, link in enumerate(link.links):
                    if isinstance(link, chainer.link.Link):
                        self.add_link("link_{}_{}".format(i,j), link)
                        #print(link.name, link)

        self.sequence = sequence
        self.test = False

    def load(self, filename):
        if os.path.isfile(filename):
            print("loading {} ...".format(filename))
            serializers.load_hdf5(filename, self)
        else:
            print(filename, "not found.")

    def save(self, filename):
        if os.path.isfile(filename):
            os.remove(filename)
        serializers.save_hdf5(filename, self)

    def get_optimizer(self, name, lr, momentum=0.9):
        if name.lower() == "adam":
            return optimizers.Adam(alpha=lr, beta1=momentum)
        if name.lower() == "smorms3":
            return optimizers.SMORMS3(lr=lr)
        if name.lower() == "adagrad":
            return optimizers.AdaGrad(lr=lr)
        if name.lower() == "adadelta":
            return optimizers.AdaDelta(rho=momentum)
        if name.lower() == "nesterov" or name.lower() == "nesterovag":
            return optimizers.NesterovAG(lr=lr, momentum=momentum)
        if name.lower() == "rmsprop":
            return optimizers.RMSprop(lr=lr, alpha=momentum)
        if name.lower() == "momentumsgd":
            return optimizers.MomentumSGD(lr=lr, mommentum=mommentum)
        if name.lower() == "sgd":
            return optimizers.SGD(lr=lr)

    def setup_optimizers(self, optimizer_name, lr, momentum=0.9, weight_decay=0, gradient_clipping=0):
        opt = self.get_optimizer(optimizer_name, lr, momentum)
        opt.setup(self)
        if weight_decay > 0:
            opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
        if gradient_clipping > 0:
            opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))
        # clip all weights to between -1 and 1
        opt.add_hook(weight_clip.WeightClip())
        self.optimizer = opt
        return self.optimizer

    def evaluate(self,x,t):
        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.sequence(*x, test=self.test)
        
        reporter.report({'numsamples': float(x[0].shape[0])}, self)
        if isinstance(self.y, tuple):
            self.loss = 0
            for i, y in enumerate(self.y):
                if isinstance(t,tuple):
                    index = min(len(t)-1,i)
                    tt = t[index]
                else:
                    tt = t
                
                branchweight = self.branchweights[min(i,len(self.branchweights)-1)]
                bloss = self.lossfun(y, tt)
                #print("branchweight",branchweight,bloss.data)
                xp = chainer.cuda.cupy.get_array_module(bloss.data)
                if y.creator is not None and not xp.isnan(bloss.data):
                    self.loss += branchweight*bloss
                reporter.report({'branch{}branchweight'.format(i): branchweight}, self)
                reporter.report({'branch{}loss'.format(i): bloss}, self)
                self.accuracy = self.accfun(y, tt)
                reporter.report({'branch{}accuracy'.format(i): self.accuracy}, self)       
            # Overall accuracy and loss of the sequence
            reporter.report({'loss': self.loss}, self)
            if self.compute_accuracy:
                y, exits = self.sequence.predict(*x, ent_Ts=self.ent_Ts, test=True)
                if isinstance(t,tuple):
                    self.accuracy = self.accfun(y, t[-1])
                else:
                    self.accuracy = self.accfun(y, t)

                reporter.report({'accuracy': self.accuracy}, self)
                for i,exit in enumerate(exits):
                    reporter.report({'branch{}exit'.format(i): float(exit)}, self)
            else:
                reporter.report({'accuracy': 0.0}, self)

            if self.ent_Ts is not None:
                #print("self.ent_Ts",self.ent_Ts)
                reporter.report({'ent_T':self.ent_Ts[0]}, self)
                
            if hasattr(self.sequence,'get_communication_costs'):
                c = self.sequence.get_communication_costs()
                reporter.report({'communication0':c[0]}, self)
                reporter.report({'communication1':c[1]}, self)
            if hasattr(self.sequence,'get_device_memory_cost'):
                reporter.report({'memory': self.sequence.get_device_memory_cost()}, self)

        return self.loss
    
    def __call__(self, *args):
        x = args[:-1]
        t = args[-1]
        return self.evaluate(x,t)
        
