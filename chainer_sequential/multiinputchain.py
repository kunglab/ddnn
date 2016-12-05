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
    
    def recursive_add_sequence(self, link, i, prefix=''):
        for j, tlink in enumerate(link.links):
            if isinstance(tlink, chainer.link.Link):
                self.add_link("{}link_{}_{}".format(prefix, i,j), tlink)
            elif isinstance(tlink, sequential.Sequential):
                self.recursive_add_sequence(tlink, str(i)+"_"+str(j))
                
    def add_sequence(self, sequence):
        if isinstance(sequence, sequential.Sequential) == False:
            raise Exception()
        if sequence.local is not None:
            link = sequence.local
            if isinstance(link, chainer.link.Link):
                self.add_link("local_link_{}".format(0), link)
            elif isinstance(link, sequential.Sequential):
                self.recursive_add_sequence(link, 0, prefix='local_')
        for i, link in enumerate(sequence.inputs):
            if isinstance(link, chainer.link.Link):
                self.add_link("input_link_{}".format(i), link)
            elif isinstance(link, sequential.Sequential):
                self.recursive_add_sequence(link, i, prefix='input_')
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
        
        return self.evaluate(x,t)