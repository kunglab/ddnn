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


#from chainer.dataset.dataset_mixin import DatasetMixin
#class Dataset(DatasetMixin):
#    def __init__(self, items):
#        self.items = items
#        self.numItems = len(items[-1])
#    def __len__(self):
#        return self.numItems
#    def get_example(self,i):
#        return tuple([item[i] for item in self.items])
    
def get_mvmc_flatten(cam=None, tr_percent=0.5):
    if cam is None:
        cam = np.arange(6).tolist()

    url = 'https://www.dropbox.com/s/uk8c6iymy8nprc0/MVMC.npz'
    base_dir = get_dataset_directory('mvmc/')
    path = os.path.join(base_dir, 'mvmc.npz')

    if not os.path.isfile(path):
        download(url, path)

    data = np.load(path)
    X = data['X']
    y = data['y']
    # Turn 3 to negative -1 for empty view
    sidx = int(len(y)*tr_percent)

    X = X[:sidx,cam]
    y = y[:sidx,cam]
    
    y = y.astype(np.int32)
    y[y==3] = -1
    
    # Get the max and
    last = np.max(y,1)
    last = last[:,np.newaxis]
    y = np.hstack([y,last])
    
    
    train_xs = X.transpose((1,0,2,3,4)).tolist()
    train_xs = [np.array(train_x).astype(np.float32) for train_x in train_xs]
    train_ys = y.transpose((1,0)).tolist()
    train_ys = [np.array(train_y).astype(np.int32) for train_y in train_ys]
    
    test_xs = X.transpose((1,0,2,3,4)).tolist()
    test_xs = [np.array(test_x).astype(np.float32) for test_x in test_xs]
    test_ys = y.transpose((1,0)).tolist()
    test_ys = [np.array(test_y).astype(np.int32) for test_y in test_ys]
    
    train = TupleDataset(*(train_xs + train_ys))
    test = TupleDataset(*(test_xs + test_ys))
    return train, test
