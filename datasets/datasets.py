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
