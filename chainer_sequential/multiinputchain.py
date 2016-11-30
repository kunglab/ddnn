import os

import numpy as np
from scipy.stats import entropy
import chainer
import sequential
import multiinputsequential
from chainer import optimizers, serializers, Variable
from chainer import functions as F
from chainer import reporter
from chainer.functions.activation.softmax import softmax
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chain import Chain

class MultiInputChain(Chain):

    def __init__(self, **kwargs):
        super(MultiInputChain,self).__init__(**kwargs)
    
    def recursive_add_sequence(self, link, i):
        for j, link in enumerate(link.links):
            if isinstance(link, chainer.link.Link):
                self.add_link("input_link_{}_{}".format(i,j), link)
            elif isinstance(link, sequential.Sequential):
                self.recursive_add_sequence(link, str(i)+"_"+str(j))
                
    def add_sequence(self, sequence):
        if isinstance(sequence, sequential.Sequential) == False:
            raise Exception()
        for i, link in enumerate(sequence.inputs):
            if isinstance(link, chainer.link.Link):
                self.add_link("input_link_{}".format(i), link)
            elif isinstance(link, sequential.Sequential):
                self.recursive_add_sequence(link, i)
        for i, link in enumerate(sequence.links):
            if isinstance(link, chainer.link.Link):
                self.add_link("link_{}".format(i), link)
                #print(link.name, link)
            elif isinstance(link, sequential.Sequential):
                self.recursive_add_sequence(link, i)

        self.sequence = sequence
        self.test = False

    def __call__(self, *args):
        split = len(args)//2
        x = args[:split]
        t = args[split:]

        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.sequence(*x, test=self.test)
        
        reporter.report({'numsamples': float(x[0].shape[0])}, self)
        if isinstance(self.y, tuple):
            self.loss = 0
            for i, y in enumerate(self.y):
                index = min(len(t)-1,i)
                #print i,len(self.y),y.data,,t[0].data
                #y = y[0]
                # TODO fix branchweight
                bloss = self.lossfun(y, t[index])
                xp = chainer.cuda.cupy.get_array_module(bloss.data)
                if y.creator is not None and not xp.isnan(bloss.data):
                    self.loss += bloss
                reporter.report({'branch{}loss'.format(i): bloss}, self)
                if self.compute_accuracy:
                    self.accuracy = self.accfun(y, t[index])
                    reporter.report({'branch{}accuracy'.format(i): self.accuracy}, self)       
            # Overall accuracy and loss of the sequence
            reporter.report({'loss': self.loss}, self)
            # TODO
            #if self.compute_accuracy:
            #    # TODO support multiple exit branch
            #    y, exited = self.sequence.predict(*x, ent_Ts=self.ent_Ts, test=self.test)
            #    numexited = float(np.sum(exited).tolist())
            #    numtotal = float(len(exited))
            #    #print("numexited",numexited)
            #    self.accuracy = self.accfun(y, t)
            #    reporter.report({'accuracy': self.accuracy}, self)
            #    reporter.report({'branch{}exit'.format(0): float(numexited)}, self)
            #    reporter.report({'branch{}exit'.format(1): float(numtotal-numexited)}, self)
            #else:
            reporter.report({'accuracy': 0.0}, self)
            reporter.report({'branch{}exit'.format(0): 0.0}, self)
            reporter.report({'branch{}exit'.format(1): 0.0}, self)
                
        return self.loss
