import chainer
import numpy as np
from chainer.datasets import TupleDataset

def get_mnist():
    train, test = chainer.datasets.get_mnist(ndim=3)
    
    train_data = [t for t in train]
    test_data = [t for t in test]
    
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    train_data = np.expand_dims(train_data, 1)
    test_data = np.expand_dims(test_data, 1)
    
    train_xs = train_data[:,:,0].T
    train_ys = train_data[:,:,1].T
    
    test_xs = test_data[:,:,0].T
    test_ys = test_data[:,:,1].T
     
    train = TupleDataset(*(train_xs.tolist() + train_ys.tolist()))
    test = TupleDataset(*(test_xs.tolist() + test_ys.tolist()))
    
    return train,test
    