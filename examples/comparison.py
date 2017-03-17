import os
import sys
sys.path.append('..')
import argparse

import chainer

from elaas.elaas import Collection
from elaas.family.simple import SimpleHybridFamily
from elaas.family.binary import BinaryFamily
from elaas.family.float import FloatFamily
from visualize import visualize
import deepopt.chooser
import matplotlib
matplotlib.rcParams['font.size'] = 20.0
import matplotlib.pyplot as plt


def max_acc(trace):
    acc = 0
    best_idx =  0
    for i, t in enumerate(trace):
        if t['action'] == 'add_point':
            acc = max(acc, t['y'])
            best_idx = i
    return acc, best_idx


model_dict = {
    "binary": BinaryFamily,
    "float": FloatFamily
    }

def train_model(args, model_type, nfilters):
    trainer = Collection(model_type, args.save_dir, nepochs=args.epochs, verbose=args.verbose)
    trainer.set_model_family(model_dict[model_type])
    train, test = chainer.datasets.get_mnist(ndim=3)
    data_shape = train._datasets[0].shape[1:]
    trainer.add_trainset(train)
    trainer.add_testset(test)
    trainer.set_searchspace(
        nfilters_embeded=[nfilters],
        nlayers_embeded=[2],
        lr=[1e-3]
    )
    res = trainer.train(niters=args.iters, bootstrap_nepochs=args.bootstrap_epochs)
    return max_acc(res)[0]



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

names = ['float', 'binary']
accs = {name: [] for name in names}
mem = {name: [] for name in names}
for i in [1,2,3,4,5]:
    acc = train_model(args, 'float', i)
    print(acc)
    accs['float'].append(acc*100)
    mem['float'].append(i*32*32*9)

print("====")

binary_accs = []
binary_mem = []
for i in [1,3,5,10,20,40,80,160]:
    acc = train_model(args, 'binary', i)
    print(acc)
    accs['binary'].append(acc*100)
    mem['binary'].append(i*32*9)


#plot code
linewidth = 4
ms = 8
colors = {'binary': '#FF944D', 'float': '#FF8F80'}
styles = {'binary': '-o', 'float': '-.o'}
plt.figure(figsize=(8, 6.5))
for name in names:
    plt.plot(mem[name], accs[name], styles[name],
             linewidth=linewidth, ms=ms, color=colors[name],
             label=name)
plt.xlabel('Memory (bits)')
plt.ylabel('Classification Accuracy (%)')
plt.legend(loc=0, prop={'size': 14})
plt.tight_layout()
plt.grid()
plt.savefig("comparison_2layer.png")
plt.clf()
