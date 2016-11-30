import numpy as np
import os
from subprocess import call
from chainer.dataset.download import get_dataset_directory
from chainer.datasets import TupleDataset


def download(url, path):
    call(['wget', url, '-O', path])

def get_mvmc(cam=None, tr_percent=0.5):
    if cam is None:
        cam = np.arange(6)

    url = 'https://www.dropbox.com/s/uk8c6iymy8nprc0/MVMC.npz'
    base_dir = get_dataset_directory('mvmc/')
    path = os.path.join(base_dir, 'mvmc.npz')

    if not os.path.isfile(path):
        download(url, path)

    data = np.load(path)
    X = data['X']
    y = data['y']
    sidx = int(len(y)*tr_percent)
    train = TupleDataset(X[:sidx, cam], y[:sidx, cam])
    test = TupleDataset(X[sidx:, cam], y[sidx:, cam])
    return train, test


def get_mvmc_flatten(cam=None, tr_percent=0.5):
    if cam is None:
        cam = np.arange(6)

    url = 'https://www.dropbox.com/s/uk8c6iymy8nprc0/MVMC.npz'
    base_dir = get_dataset_directory('mvmc/')
    path = os.path.join(base_dir, 'mvmc.npz')

    if not os.path.isfile(path):
        download(url, path)

    data = np.load(path)
    X = data['X']
    y = data['y']
    # Turn 3 to negative -1 for empty view
    y = y.astype(np.int32)
    y[y==3] = -1
    
    # Get the max and
    last = np.max(y,1)
    last = last[:,np.newaxis]
    y = np.hstack([y,last])
    
    sidx = int(len(y)*tr_percent)
    
    train_xs = X[:sidx].transpose((1,0,2,3,4)).tolist()
    train_ys = y[:sidx].transpose((1,0)).tolist()
    
    test_xs = X[sidx:].transpose((1,0,2,3,4)).tolist()
    test_ys = y[sidx:].transpose((1,0)).tolist()
    
    train = TupleDataset(*(train_xs + train_ys))
    test = TupleDataset(*(test_xs + test_ys))
    return train, test
