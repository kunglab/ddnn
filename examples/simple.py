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

parser = argparse.ArgumentParser(description='Training Simple eBNN model')
parser.add_argument('-s', '--save_dir', default='_models')
parser.add_argument('-c', '--c_file', default=os.path.join('c', 'simple.h'))
parser.add_argument('--inter_file', default=os.path.join('c', 'inter_simple.h'))
parser.add_argument('-i', '--iters', type=int, default=10)
parser.add_argument('-e', '--epochs', type=int,  default=20)
parser.add_argument('-b', '--bootstrap_epochs', type=int,  default=2)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('--gen_inter', action='store_true')
args = parser.parse_args()

#Setup model type (e.g. Binary)
trainer = Collection('binary', args.save_dir, nepochs=args.epochs, verbose=args.verbose)
trainer.set_model_family(BinaryFamily)

#Dataset
train, test = chainer.datasets.get_mnist(ndim=3)
data_shape = train._datasets[0].shape[1:]
trainer.add_trainset(train)
trainer.add_testset(test)

#Model parameters
trainer.set_searchspace(
    nfilters_embeded=[2],
    nlayers_embeded=[1],
    lr=[1e-3]
)

#Train model
trainer.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)

# generate eBNN C library
trainer.generate_c(args.c_file, data_shape)
if args.gen_inter:
    inter = trainer.generate_inter_results_c(args.inter_file, train._datasets[0][:1])
