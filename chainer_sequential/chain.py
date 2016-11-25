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
                 branchweight=1, ent_T=None,
                 accfun=accuracy.accuracy):
        super(Chain,self).__init__()
        self.lossfun = lossfun
        self.branchweight = branchweight
        self.ent_T = ent_T
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

    def ent_acc(self, t):
        '''
        Entropy based accuracy for single branch case
        '''
        #exit all at last branch if not set
        if self.ent_T is None:
            return self.accfun(self.y[0], t)

        y_last, y_br = self.y
        y_last.to_cpu()
        y_br.to_cpu()
        t.to_cpu()

        y_br_sm = softmax(y_br)
        y_br_ent = entropy(y_br_sm.data.T)
        idxs = y_br_ent < self.ent_T
        num_early = np.sum(idxs)
        num_last = np.sum(~idxs)

        y_early_exit = Variable(y_br.data[idxs==True])
        t_early = Variable(t.data[idxs==True])
        y_last_exit = Variable(y_last.data[idxs==False])
        t_last = Variable(t.data[idxs==False])
        y_early_exit.to_gpu()
        y_last_exit.to_gpu()
        t_early.to_gpu()
        t_last.to_gpu()

        #if no samples exit early, set acc to 1.
        if num_early == 0:
            early_acc = 1
        else:
            early_acc = self.accfun(y_early_exit, t_early).data
        last_acc = self.accfun(y_last_exit, t_last).data
        acc = (early_acc*num_early + last_acc*num_last) / (num_early + num_last)

        y_last.to_gpu()
        y_br.to_gpu()
        t.to_gpu()

        return acc

    def __call__(self, *args):
        x = args[:-1]
        t = args[-1]

        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.sequence(*x, test=self.test)

        if isinstance(self.y, tuple):
            self.loss = self.lossfun(self.y[0], t)
            for i, y in enumerate(self.y[1:]):
                self.branchloss = self.branchweight*self.lossfun(y, t)
                self.loss += self.branchweight*self.branchloss
                reporter.report({'branch{}loss'.format(i): self.branchloss}, self)
                if self.compute_accuracy:
                    self.accuracy = self.accfun(y, t)
                    reporter.report({'branch{}accuracy'.format(i): self.accuracy}, self)
            reporter.report({'loss': self.loss}, self)
            if self.compute_accuracy:
                #TODO: replace here
                self.accuracy = self.ent_acc(t)
                reporter.report({'accuracy': self.accuracy}, self)
        else:
            self.loss = self.lossfun(self.y, t)
            reporter.report({'loss': self.loss}, self)
            if self.compute_accuracy:
                self.accuracy = self.accfun(self.y, t)
                reporter.report({'accuracy': self.accuracy}, self)

        return self.loss
