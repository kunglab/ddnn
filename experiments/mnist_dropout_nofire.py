from __future__ import absolute_import
from __future__ import print_function

import random
random.seed(9001)

import os
import sys
sys.path.append('..')
import argparse

import chainer
import numpy as np
#from datasets.datasets import get_mvmc, get_mvmc_flatten
from datasets.mnist import get_mnist

import deepopt.chooser
from elaas.elaas import Collection
from elaas.family.multi_input_edge_with_dropout import MultiInputEdgeDropoutFamily
from visualize import visualize
import matplotlib
matplotlib.rcParams['font.size'] = 20.0
import matplotlib.pyplot as plt

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def run(dtrain, dtest, epochs=10, verbose=False):
    args = AttrDict(**dict(
        save_dir="_models",
        iters=epochs,
        epochs=epochs,
        bootstrap_epochs=1,
        ncams=1,
        verbose=verbose
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
        nfilters_embeded=[3],
        nlayers_embeded=[2],
        nfilters_cloud=[3],
        nlayers_cloud=[2],
        lr=[1e-3],
        branchweight=[.1],
        ent_T=[100]
    )

    # currently optimize based on the validation accuracy of the main model
    traces = mnist.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
    return traces[-1]['y']

dtrains = {
    'no_d': 0.0,
    '50_d': 0.5
    }
dtests = [0.0, 0.1, 0.25, 0.5, 0.75]

acc_dict = {}
for key, dtrain in dtrains.iteritems():
    acc_dict[key] = []
    for dtest in dtests:
        print(dtrain, dtest)
        y = run(dtrain, dtest, 10)
        print(y)
        acc_dict[key].append(y*100.)


#plot code
names = ['no_d', '50_d']
linewidth = 4
ms = 8
colors = {'no_d': '#F93943', '50_d': '#445E93'}
styles = {'no_d': '-o', '50_d': '--o'}
labels = {'no_d': 'No Dropout', '50_d': 'Dropout (50%)'}
plt.figure(figsize=(8, 6.5))
comms = (1-np.array(dtests))*100.
for name in names:
    plt.plot(comms, acc_dict[name], styles[name],
             linewidth=linewidth, ms=ms, color=colors[name],
             label=labels[name])
plt.xlabel('Communication (%)')
plt.ylabel('Classification Accuracy (%)')
plt.legend(loc=0, prop={'size': 14})
plt.tight_layout()
plt.grid()
plt.savefig("dropout.png")
plt.clf()
