import os
import sys
sys.path.append('..')
import argparse

import chainer

from elaas.elaas import Collection
from elaas.family.simple import SimpleHybridFamily
from visualize import visualize
import deepopt.chooser

parser = argparse.ArgumentParser(description='Hybrid Example')
parser.add_argument('-s', '--save_dir', default='_models')
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('-e', '--epochs', type=int,  default=20)
parser.add_argument('-b', '--bootstrap_epochs', type=int,  default=2)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

train, test = chainer.datasets.get_mnist(ndim=3)


# binary = Collection('simple_binary', args.save_dir, nepochs=args.epochs, verbose=args.verbose)
# binary.set_model_family(SimpleFamily)
# binary.add_trainset(train)
# binary.add_testset(test)
# binary.set_searchspace(
#     nfilters_embeded=[1],
#     nlayers_embeded=[2],
#     nfilters_cloud=[1],
#     nlayers_cloud=[1],
#     lr=[1e-3],
#     branchweight=[.1],
#     ent_T=[0.0001, 0.001, 0.005, 0.01, 0.1]
# )
# binary.set_chooser(deepopt.chooser.GridChooser())

hybrid = Collection('simple_hybrid', args.save_dir, nepochs=args.epochs, verbose=args.verbose)
hybrid.set_model_family(SimpleHybridFamily)
hybrid.add_trainset(train)
hybrid.add_testset(test)
hybrid.set_searchspace(
    nfilters_embeded=[1],
    nlayers_embeded=[2],
    nfilters_cloud=[1],
    nlayers_cloud=[1],
    lr=[1e-3],
    branchweight=[.1],
    ent_T=[0.0001, 0.001, 0.005, 0.01, 0.1, 100]
)
hybrid.set_chooser(deepopt.chooser.GridChooser())



# binary_traces = binary.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
hybrid_traces = hybrid.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
