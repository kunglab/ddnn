import warnings
import numpy as np
import scipy.stats as sps

def get_max_epoch(do, new_point):
    X = np.array(do.X)
    for i, param in enumerate(do.params):
        if param == 'nepochs':
            continue
        X = X[X[:, i] == new_point[param]]

    return np.max(X[:, do.params.index('nepochs')])

def compute_ts(do):
    X_samples = np.array(do.X_samples)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        y_pred, sigma = do.gp.predict(X_samples, return_std=True)
    X = np.array(do.X)

    ts = []
    for x in X_samples:
        x = dict(zip(do.params, x))
        epochs = []
        for xi in X:
            xi = dict(zip(do.params, xi))
            same_model = True
            for k in x.keys():
                if x[k] != xi[k] and k != 'nepochs':
                    same_model = False

            if same_model:
                epochs.append(xi['nepochs'])

        start_epoch = np.max(epochs)
        t = max(1, x['nepochs'] - start_epoch)
        ts.append(t)
    ts = np.array(ts)

    return ts



class SimpleChooser(object):
    def __init__(self, beta=1):
        self.beta = beta
    def choose(self, do, y_pred, sigma):
        return np.argmax(u+self.beta*v)

class EIChooser(object):
    def __init__(self):
        pass

    def get_ei(self, do, y_pred, sigma, eps=1e-9, ts=None):
        ei_values = np.min(do.y, axis=0)
        y_pred = np.expand_dims(y_pred,1)
        sigma = np.expand_dims(sigma,1)
        func_s = sigma
        u      = (ei_values - y_pred) / (func_s+eps)
        ncdf   = sps.norm.cdf(u)
        npdf   = sps.norm.pdf(u)
        ei     = np.mean(func_s*( u*ncdf + npdf),axis=1)

        if ts is not None:
            ei = ei / ts

        return ei

    def choose(self, do, y_pred, sigma, ts=None, norm_k=1, eps=1e-9, return_values=False):
        ei = self.get_ei(do, y_pred, sigma, eps, ts)
        max_idx = np.argmax(ei)
        if return_values:
            return max_idx, ei
        else:
            return max_idx


class GridChooser(object):
    def __init__(self):
        self.init = False

    def choose_gen(self, do, y_pred, sigma, return_values=False):
        X_samples = np.array(do.X_samples)
        epoch_idx = do.params.index('nepochs')
        max_epoch = np.max(X_samples[:, epoch_idx])
        while True:
            for idx, x in enumerate(do.X_samples):
                if x[epoch_idx] == max_epoch:
                    if return_values:
                        yield idx, np.zeros(len(y_pred))
                    else:
                        yield idx

    def choose(self, do, y_pred, sigma, return_values=False):
        if not self.init:
            self.gen = self.choose_gen(do, y_pred, sigma, return_values=return_values)
            self.init = True

        return self.gen.next()


class EpochChooser(EIChooser):
    def __init__(self, k=5, **kwargs):
        super(EpochChooser, self).__init__(**kwargs)
        self.k = k

    def choose(self, do, y_pred, sigma, return_values=False):
        traces = do.get_traces()
        traces = [t for t in traces if t['action'] == 'add_point']
        if len(traces) == 0:
            curr_point = None
        else:
            curr_point = dict(zip(do.params, traces[-1]['x']))

        X_samples = np.array(do.X_samples)
        X = np.array(do.X)
        ts = compute_ts(do)
        ei = self.get_ei(do, y_pred, sigma, ts=ts)
        ei = np.array(ei)
        top_idxs = np.argsort(ei)[::-1][:self.k]
        max_epoch, best_idx = 0, 0
        for idx in top_idxs:
            new_point = dict(zip(do.params, X_samples[idx]))
            if curr_point is not None:
                same_model = True
                for key in new_point.keys():
                    if curr_point[key] != new_point[key] and key != 'nepochs':
                        same_model = False

                if same_model:
                    best_idx = idx
                    break

            epoch = get_max_epoch(do, new_point)
            if epoch > max_epoch:
                best_idx = idx

        if return_values:
            return best_idx, ei
        else:
            return best_idx
