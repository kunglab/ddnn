import os

import numpy as np
import matplotlib.pyplot as plt


def get_max_epoch(do, new_point):
    X = np.array(do.X)
    for i, param in enumerate(do.params):
        if param == 'nepochs':
            continue
        X = X[X[:, i] == new_point[param]]

    return np.max(X[:, do.params.index('nepochs')])



def embed_transmit_err(do, traces, save_dir):
    save_dir = os.path.join(save_dir, 'figures/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nfilters_idx = do.params.index('nfilters_embeded')
    nlayers_idx = do.params.index('nlayers_embeded')

    xs, ys = [], []
    model_hashes = []
    for t in traces[::-1]:
        if t['action'] != 'add_point':
            continue
        point = dict(zip(do.params, t['x']))
        point.pop('nepochs', None)
        h = hash(frozenset(point.items()))
        if h not in model_hashes:
            model_hashes.append(h)
            xs.append((point['nfilters_embeded']**2)*(9/8))
            ys.append(t['y'])


    plt.figure(figsize=(8, 6.5))
    plt.plot(xs, ys, 'o')
    x_rng = (np.max(xs) - np.min(xs))*0.05
    y_rng = (np.max(ys) - np.min(ys))*0.05
    plt.xlim((np.min(xs)-x_rng, np.max(xs)+x_rng))
    plt.ylim((np.min(ys)-y_rng, np.max(ys)+y_rng))
    plt.xlabel('transmission size (bytes)')
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.savefig(save_dir + 'transmit_error.png')
    plt.clf()


def embed_memory_err(do, traces, save_dir):
    save_dir = os.path.join(save_dir, 'figures/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nfilters_idx = do.params.index('nfilters_embeded')
    nlayers_idx = do.params.index('nlayers_embeded')

    xs, ys = [], []
    model_hashes = []
    for t in traces[::-1]:
        if t['action'] != 'add_point':
            continue
        point = dict(zip(do.params, t['x']))
        point.pop('nepochs', None)
        h = hash(frozenset(point.items()))
        if h not in model_hashes:
            model_hashes.append(h)
            xs.append((point['nfilters_embeded']**2)*point['nlayers_embeded']*(9/8))
            ys.append(t['y'])

    plt.figure(figsize=(8, 6.5))
    plt.plot(xs, ys, 'o')
    plt.xlabel('embedded memory size (bytes)')
    plt.ylabel('accuracy')
    x_rng = (np.max(xs) - np.min(xs))*0.05
    y_rng = (np.max(ys) - np.min(ys))*0.05
    plt.xlim((np.min(xs)-x_rng, np.max(xs)+x_rng))
    plt.ylim((np.min(ys)-y_rng, np.max(ys)+y_rng))
    plt.tight_layout()
    plt.savefig(save_dir + 'memory_error.png')
    plt.clf()

def min_error(traces, save_dir):
    save_dir = os.path.join(save_dir, 'figures/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    points = [t for t in traces
              if t['action'] == 'add_point']
    plt.figure(figsize=(8, 6.5))
    ys = [1-p['y'] for p in points]
    plt.plot(np.minimum.accumulate(ys), linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel('minimum error')
    plt.tight_layout()
    plt.savefig(save_dir + 'gp_min_error.png')
    plt.clf()
