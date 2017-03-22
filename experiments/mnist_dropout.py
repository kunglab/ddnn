from __future__ import absolute_import

import random
random.seed(9001)

import os
import sys
sys.path.append('..')
import argparse

import chainer
#from datasets.datasets import get_mvmc, get_mvmc_flatten
from datasets.mnist import get_mnist

import deepopt.chooser
from elaas.elaas import Collection
from elaas.family.multi_input_edge_with_dropout import MultiInputEdgeDropoutFamily
from visualize import visualize
import fire

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class MNISTDropout(object):
    def run(dtrain, dtest):
        args = AttrDict(**dict(
            save_dir="_models",
            iters=1,
            epochs=2,
            bootstrap_epochs=2,
            ncams=1,
            verbose=True
        ))

        mnist = Collection('multiinput_edge_dropout_mpcc_{}'.format(args.ncams), args.save_dir, nepochs=args.epochs, verbose=args.verbose)
        ncams = args.ncams
        mnist.set_model_family(MultiInputEdgeDropoutFamily, ninputs=ncams, resume=False,
                               merge_function="max_pool_concat", drop_comm_train=dtrain,
                               drop_comm_test=dtest, input_dims=1, output_dims=10)
        train, test = get_mnist()
        mnist.add_trainset(train)
        mnist.add_testset(test)
        mnist.set_searchspace(
            nfilters_embeded=[4],
            nlayers_embeded=[2],
            nfilters_cloud=[4],
            nlayers_cloud=[2],
            lr=[1e-3],
            branchweight=[.1],
            ent_T=[100]
        )

        # switch chooser
        # mnist.set_chooser(deepopt.chooser.EpochChooser(k=5))

        # currently optimize based on the validation accuracy of the main model
        traces = mnist.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
        return traces.y


if __name__ == '__main__':
    fire.Fire(MNISTDropout)
