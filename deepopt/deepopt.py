import warnings
from itertools import product
from collections import Iterable
from time import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from .chooser import EIChooser

def cross(old_items,new_item):
    old_items = np.array(old_items)
    old_length = old_items.shape[0]
    items = []
    for i in range(old_items.shape[1]):
        data = old_items[:,i]
        items.append(np.repeat([data], len(new_item), axis=0).ravel())
    items.append(np.repeat([new_item], old_length, axis=1).ravel())
    return zip(*items)

class DeepOptEpoch(object):
    def __init__(self, nepochs=10, kernel=RBF(5, (1, 10)), n_restarts_optimizer=3):
        self.params = []
        self.X_samples = [[]]
        self.X = []
        self.y = []
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        self.nepochs = nepochs
        self.add_param('nepochs', range(1,self.nepochs+1))
        self.traces = None
        self.constraintfn = None
        self.chooser = EIChooser()

    def set_chooser(self, chooser=EIChooser()):
        self.chooser = chooser

    def set_constraints(self, constraintfn):
        self.constraintfn = constraintfn
        X_samples = []
        for x in self.X_samples:
            if constraintfn(**self.point_to_kwargs(x)):
                X_samples.append(x)
        self.X_samples = X_samples

    def add_param(self, name, values):
        self.params.append(name)
        self.X_samples = cross(self.X_samples, values)

    def add_points(self, epochs, ys, **kwargs):
        for j in range(len(epochs)):
            self.add_point(epochs[j], ys[j], **kwargs)

    def add_point(self, epoch, y, **kwargs):
        kwargs['nepochs'] = epoch
        point = []
        for k,v in kwargs.iteritems():
            dim = self.params.index(k)
            point.append((dim,v))
        x = [d[1] for d in sorted(point)]
        self.X.append(x)
        self.y.append(y)

    def save(self, fn):
        np.savez(fn, params=self.params,X_samples=self.X_samples,
                 X=self.X,y=self.y,nepochs=self.nepochs)

    def load(self, fn):
        data = np.load(fn)
        self.params = data['params'].tolist()
        self.X_samples = data['X_samples'].tolist()
        self.X = data['X'].tolist()
        self.y = data['y'].tolist()
        self.nepochs = int(data['nepochs'].tolist())

    # deprecated
    def add_sample_points(self, **kwargs):
        keys = kwargs.keys()
        items = list(kwargs.iteritems())
        for i in range(1,self.nepochs+1):
            for j in range(len(keys[0])):
                point = []
                point.append((0,i)) # nepochs
                for k,v in items:
                    dim = self.params.index(k)
                    point.append((dim,v[j]))
                x = [d[1] for d in sorted(point)]
                self.X_samples.append(x)

    # deprecated
    def add_sample_point(self, **kwargs):
        for i in range(1,self.nepochs+1):
            point = []
            point.append((0,i)) # nepochs
            for k,v in kwargs.iteritems():
                dim = self.params.index(k)
                point.append((dim,v))
            x = [d[1] for d in sorted(point)]
            self.X_samples.append(x)

    def fit(self):
        X = np.array(self.X)
        y = np.array(self.y)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.gp.fit(X,y)

    def start_traces(self):
        self.traces = []

    def sample_point(self):
        chooser = self.chooser

        X_samples = np.array(self.X_samples)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            y_pred, sigma = self.gp.predict(X_samples, return_std=True)

        if self.traces is not None:
            max_idx, chooser_values = chooser.choose(self, y_pred, sigma, return_values=True)
        else:
            max_idx = chooser.choose(self, y_pred, sigma)
        point = X_samples[max_idx]
        max_point = {}
        for k,v in enumerate(point):
            max_point[self.params[k]] = v

        if self.traces is not None:
            self.traces.append(dict(X_samples=X_samples,
                                    y_pred=y_pred,
                                    sigma=sigma,
                                    max_idx=max_idx,
                                    max_point=max_point,
                                    chooser_values=chooser_values))

        return max_point

    def stop_traces(self):
        traces = self.traces
        self.traces = None
        return traces

    def point_to_kwargs(self, point):
        data = {}
        for k,v in enumerate(point):
            data[self.params[k]] = v
        return data

    def get_bootstrap_points(self, bootstrap_nepochs=2):
        points = [self.point_to_kwargs(point) for point in self.X_samples if point[0] == bootstrap_nepochs]
        return points

    def get_ys(self, **kwargs):
        data = []
        for i in range(1,self.nepochs+1):
            kwargs['nepochs'] = i
            data.append(self.get_y(**kwargs))
        return np.hstack(data)

    def get_y(self, **kwargs):
        point = []
        for k,v in kwargs.iteritems():
            dim = self.params.index(k)
            point.append((dim,v))
        x = [d[1] for d in sorted(point)]
        idx = self.X_samples.index(tuple(x))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            y_pred, sigma = self.gp.predict(self.X_samples[idx:idx+1], return_std=True)
        return y_pred

    def get_best_point(self):
        idx = np.argmin(self.y)
        point = self.X[idx]
        data = {}
        for k,v in enumerate(point):
            data[self.params[k]] = v
        return data
