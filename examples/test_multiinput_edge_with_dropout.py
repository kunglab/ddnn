import os
import sys
sys.path.append('..')
import argparse

import chainer
from datasets.datasets import get_mvmc, get_mvmc_flatten
import deepopt.chooser
from elaas.elaas import Collection
from elaas.family.multi_input_edge_with_dropout import MultiInputEdgeDropoutFamily
from visualize import visualize

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
parser.add_argument('-e', '--epochs', type=int,  default=40)
parser.add_argument('-b', '--bootstrap_epochs', type=int,  default=2)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-n', '--ncams', type=int,  default=6)
args = parser.parse_args()

mnist = Collection('multiinput_edge_dropout_mpcc_{}'.format(args.ncams), args.save_dir, nepochs=args.epochs, verbose=args.verbose)

ncams = args.ncams
mnist.set_model_family(MultiInputEdgeDropoutFamily,ninputs=ncams,resume=False,merge_function="max_pool_concat",drop_comm_train=0.5,drop_comm_test=0.5, input_dims=3)

# train, test = get_mvmc_flatten(range(ncams))
train, test = chainer.datasets.get_cifar10(ndim=3)
# train, test = chainer.datasets.get_mnist(ndim=3)
# assert False

#from chainer.datasets.sub_dataset import SubDataset
#train = SubDataset(train, 0, 500)
#test = SubDataset(train, 0, 500)

mnist.add_trainset(train)
mnist.add_testset(test)

mnist.set_searchspace(
    nfilters_embeded=[32],
    nlayers_embeded=[2],
    nfilters_cloud=[32],
    nlayers_cloud=[2],
    lr=[1e-3],
    branchweight=[.1],
    ent_T=[100]
)

def constraintfn(**kwargs):
    #TODO: change to memory cost
    return True

mnist.set_constraints(constraintfn)

# switch chooser
mnist.set_chooser(deepopt.chooser.EpochChooser(k=5))

# currently optimize based on the validation accuracy of the main model
traces = mnist.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
#idx, acc = max_acc(traces)
#print('Best Binary Acc: {:2.4f}'.format(acc*100.))
