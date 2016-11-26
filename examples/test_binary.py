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

parser = argparse.ArgumentParser(description='Hybrid Example')
parser.add_argument('-s', '--save_dir', default='_models')
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('-e', '--epochs', type=int,  default=20)
parser.add_argument('-b', '--bootstrap_epochs', type=int,  default=2)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

mnist = Collection('binary', args.save_dir, nepochs=args.epochs, verbose=args.verbose)
mnist.set_model_family(BinaryFamily)

train, test = chainer.datasets.get_mnist(ndim=3)
mnist.add_trainset(train)
mnist.add_testset(test)

# mnist.set_searchspace(
#     nfilters_embeded=[5, 10],
#     nlayers_embeded=[1, 2],
#     nfilters_cloud=[5],
#     nlayers_cloud=[1],
#     lr=[1e-3],
#     branchweight=[.1],
#     ent_T=[0.1, 0.2, 0.4]
# )

mnist.set_searchspace(
    nfilters_embeded=[4, 8, 16],
    nlayers_embeded=[2],
    nfilters_cloud=[16],
    nlayers_cloud=[3],
    lr=[1e-3, 1e-4],
    branchweight=[.1],
    ent_T=[0.5, 0.6]
)

def constraintfn(**kwargs):
    #TODO: change to memory cost
    return True

mnist.set_constraints(constraintfn)

# switch chooser
mnist.set_chooser(deepopt.chooser.EpochChooser(k=5))

# currently optimize based on the validation accuracy of the main model
traces = mnist.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
visualize.min_error(traces, args.save_dir)
visualize.embed_memory_err(mnist.do, traces, args.save_dir)
visualize.embed_transmit_err(mnist.do, traces, args.save_dir)

# generate c
# mnist.generate_c((1,28,28))

# generate container
# mnist.generate_container()

# get traces for the collection
# mnist = Collection('simple_hybrid', save_dir)
# traces = mnist.get_do_traces()
