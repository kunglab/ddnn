import numpy as np
import scipy.stats as sps

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
    
# TODO: ADD OUR CHOOSER HERE. KEEP STATE INSIDE CHOOSER. PASSINGIN PARAMS FROM OUTSIDE AS LITTLE AS POSSIBLE 

# TODO: ADD GRID SEARCH CHOOSER

# TODO: ADD NP EPOCH GP CHOOSER etc...