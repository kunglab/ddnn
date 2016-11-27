from __future__ import print_function
import os
import sys
sys.path.append('..')
import argparse

import chainer

from elaas.elaas import Collection
from elaas.family.simple import SimpleHybridFamily
from elaas.family.binary import BinaryFamily
from visualize import visualize
import deepopt.chooser

def max_acc(trace):
    acc = 0
    best_idx =  0
    for i, t in enumerate(trace):
        if t['action'] == 'add_point':
            acc = max(acc, t['y'])
            best_idx = i
    return acc, best_idx

parser = argparse.ArgumentParser(description='Hybrid Example')
parser.add_argument('-s', '--save_dir', default='_models')
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('-e', '--epochs', type=int,  default=20)
parser.add_argument('-b', '--bootstrap_epochs', type=int,  default=2)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

train, test = chainer.datasets.get_mnist(ndim=3)


hybrid = Collection('simple_hybrid', args.save_dir, nepochs=args.epochs, verbose=args.verbose)
hybrid.set_model_family(SimpleHybridFamily)
hybrid.add_trainset(train)
hybrid.add_testset(test)
hybrid.set_searchspace(
    nfilters_embeded=[64],
    nlayers_embeded=[2],
    nfilters_cloud=[64],
    nlayers_cloud=[2],
    lr=[1e-3],
    branchweight=[.5],
    ent_T=[0., 0.005, 0.001, 0.1, 100]
)
hybrid.set_chooser(deepopt.chooser.GridChooser())



binary_traces = binary.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
idx, acc = max_acc(binary_traces)
print('Best Binary Acc: {:2.4f}'.format(acc*100.))
