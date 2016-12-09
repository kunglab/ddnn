import os
import sys
sys.path.append('..')
import argparse

import chainer
from datasets.datasets import get_mvmc, get_mvmc_flatten_eval
import deepopt.chooser
from elaas.elaas import Collection
from elaas.family.multi_input_edge import MultiInputEdgeFamily
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
parser.add_argument('-e', '--epochs', type=int,  default=100)
parser.add_argument('-b', '--bootstrap_epochs', type=int,  default=100)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-n', '--ncams', type=str,  default="0,1,2,3,4,5")
args = parser.parse_args()

args.ncams = [int(i) for i in args.ncams.split(",")]
mnist = Collection('exp_model_size_{}'.format(reduce(lambda x,y: str(x)+str(y), args.ncams)), args.save_dir, input_dims=3, nepochs=args.epochs, verbose=args.verbose)

ncams = args.ncams
mnist.set_model_family(MultiInputEdgeFamily,ninputs=len(ncams),batchsize=700,merge_function="max_pool_concat")

train, test = get_mvmc_flatten_eval(ncams)
#print("train",len(train))
#print("test",len(test))
#import sys
#sys.exit(0)
#train, test = chainer.datasets.get_cifar10(ndim=3)
#print(train[1])
#print(len(train[0]))
#from chainer.datasets.sub_dataset import SubDataset
#train = SubDataset(train, 0, 500)
#test = SubDataset(train, 0, 500)

mnist.add_trainset(train)
mnist.add_testset(test)

mnist.set_searchspace(
    nfilters_embeded_last=[1],
    nfilters_embeded=[1,2,3,4,8,16],
    nlayers_embeded=[1],
    #nfilters_edge=[16],
    #nlayers_edge=[2],
    nfilters_cloud=[16],
    nlayers_cloud=[2],
    lr=[1e-3],
    branchweight=[1],
    ent_T=[0.8]
)

#mnist.set_searchspace(
#    nfilters_embeded_last=[8],
#    nfilters_embeded=[8],
#    nlayers_embeded=[2],
#    nfilters_edge=[16],
#    nlayers_edge=[1],
#    nfilters_cloud=[16],
#    nlayers_cloud=[2],
#    lr=[1e-3],
#    branchweight=[1],
#    ent_T=[0.8]
#)

# mnist.set_searchspace(
#     nfilters_embeded=[64, 128],
#     nlayers_embeded=[1, 2],
#     nfilters_cloud=[128],
#     nlayers_cloud=[2,4],
#     lr=[1e-3,1e-4, 1e-5],
#     branchweight=[.1],
#     ent_T=[0.0, 0.01, 0.1]
# )

def constraintfn(**kwargs):
    #TODO: change to memory cost
    return True

mnist.set_constraints(constraintfn)

# switch chooser
mnist.set_chooser(deepopt.chooser.GridChooser())

# currently optimize based on the validation accuracy of the main model
traces = mnist.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
#idx, acc = max_acc(traces)
#print('Best Binary Acc: {:2.4f}'.format(acc*100.))

# visualize.min_error(traces, args.save_dir)
# visualize.embed_memory_err(mnist.do, traces, args.save_dir)
# visualize.embed_transmit_err(mnist.do, traces, args.save_dir)

# generate c
# mnist.generate_c((1,28,28))

# generate container
# mnist.generate_container()

# get traces for the collection
# mnist = Collection('simple_hybrid', save_dir)
# traces = mnist.get_do_traces()
