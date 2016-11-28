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

class Chain(chainer.Chain):
    compute_accuracy = True

    def __init__(self, lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 branchweight=1, ent_T=None, ent_Ts=None,
                 accfun=accuracy.accuracy):
        super(Chain,self).__init__()
        self.lossfun = lossfun
        self.branchweight = branchweight
        #self.ent_T = ent_T
        if ent_T is not None and ent_Ts is None:
            self.ent_Ts = [ent_T]
        else:
            self.ent_Ts = ent_Ts
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def add_sequence(self, sequence):
        if isinstance(sequence, sequential.Sequential) == False:
            raise Exception()
        for i, link in enumerate(sequence.links):
            if isinstance(link, chainer.link.Link):
                self.add_link("link_{}".format(i), link)
            elif isinstance(link, sequential.Sequential):
                for j, link in enumerate(link.links):
                    if isinstance(link, chainer.link.Link):
                        self.add_link("link_{}_{}".format(i,j), link)

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
        self.optimizer = opt
        return self.optimizer

    def __call__(self, *args):
        x = args[:-1]
        t = args[-1]

        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.sequence(*x, test=self.test)

        reporter.report({'numsamples': float(x[0].shape[0])}, self)
        if isinstance(self.y, tuple):
            self.loss = 0
            for i, y in enumerate(self.y):
                #y = y[0]
                #print(y.shape)
                bloss = self.branchweight*self.lossfun(y, t)
                xp = chainer.cuda.cupy.get_array_module(bloss.data)
                if not xp.isnan(bloss.data):
                    self.loss += self.branchweight*bloss
                reporter.report({'branch{}loss'.format(i): bloss}, self)
                if self.compute_accuracy:
                    self.accuracy = self.accfun(y, t)
                    reporter.report({'branch{}accuracy'.format(i): self.accuracy}, self)       
            # Overall accuracy and loss of the sequence
            reporter.report({'loss': self.loss}, self)
            if self.compute_accuracy:
                y, exited = self.sequence.predict(*x, ent_Ts=self.ent_Ts, test=self.test)
                numexited = float(np.sum(exited).tolist())
                numtotal = float(len(exited))
                #print("numexited",numexited)
                self.accuracy = self.accfun(y, t)
                reporter.report({'accuracy': self.accuracy}, self)
                reporter.report({'branch{}exit'.format(0): numexited}, self)
                reporter.report({'branch{}exit'.format(1): numtotal-numexited}, self)
                
        else:
            self.loss = self.lossfun(self.y, t)
            reporter.report({'loss': self.loss}, self)
            if self.compute_accuracy:
                self.accuracy = self.accfun(self.y, t)
                reporter.report({'accuracy': self.accuracy}, self)

        return self.loss
