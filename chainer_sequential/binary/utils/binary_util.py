import numpy as np

storage_ts = ['row_major', 'col_major']

def binarize(W):
    return np.where(W>=0, 1, -1).astype(np.float32, copy=False)

def binarize_real(W):
    return np.where(W>=0, 1, 0).astype(int, copy=False)


def binary_linear(x, W, b):
    Wb = binarize(W)
    y = x.dot(Wb.T)
    y += b

    return y

'''
X: input
F: filters
m: dim 0 (# samples)
n: dim 1 (# channels)
w: dim 2 (width)
h: dim 3 (height)
num_f: # filters
kw: filter width
kh: filter height
sw: stride width
sh: stride height
'''
def conv(X, F, sw=1, sh=1):
    m = X.shape[0]
    n = X.shape[1]
    num_f = F.shape[0]
    w, h = X.shape[2], X.shape[3]
    kw, kh = F.shape[2], F.shape[3]
    ret = np.zeros((m, num_f, w - kw + 1, h - kh + 1))
    mags = np.zeros(F.shape[:2])
    for i in range(0, m):
        for j in range(0, n):
            for f in range(0, num_f):
                fi = F[f, j].reshape(-1, 1)
                for p in range(0, w-kw+1, sw):
                    for q in range(0, h-kh+1, sh):
                        x = X[i, j, p:p+kw, q:q+kh].reshape(1, -1)
                        res = np.dot(x, fi)[0,0]
                        ret[i, f, p, q] += np.dot(x, fi)[0,0]
                        mags[f, j] += np.abs(res)

    return ret, mags



def batch_norm(x, gamma, beta, mean, var):
    head_ndim = gamma.ndim + 1
    expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
    gamma = gamma[expander]
    beta = beta[expander]

    x = x - mean[expander]
    std = np.sqrt(var, dtype=var.dtype)
    x = x / std[expander]
    y = gamma * x
    y += beta

    return y


def forward_pass(x, mlp):
    x = binary_linear(x, mlp.l1.W.data, mlp.l1.b.data)
    x = batch_norm(x, mlp.b1.gamma.data, mlp.b1.beta.data, mlp.b1.avg_mean, mlp.b1.avg_var)
    x = binarize(x)

    x = binary_linear(x, mlp.l2.W.data, mlp.l2.b.data)
    x = batch_norm(x, mlp.b2.gamma.data, mlp.b2.beta.data, mlp.b2.avg_mean, mlp.b2.avg_var)
    x = binarize(x)

    x = binary_linear(x, mlp.l3.W.data, mlp.l3.b.data)
    x = batch_norm(x, mlp.b3.gamma.data, mlp.b3.beta.data, mlp.b3.avg_mean, mlp.b3.avg_var)

    return x



def np_to_floatC(xs, name, storage_t):
    '''
    Converts a numpy array into a float C array
    '''
    if storage_t not in storage_ts:
        print('storage_t: {} invalid. Options are {}.'.format(storage_t, storage_ts))
        return

    if storage_t == 'col_major':
        xs = xs.T

    xs = xs.flatten()
    float_buf = []
    float_buf = list(map(repr, xs))

    c_str = 'float {}[{}] = {{{}}};'.format(name, len(float_buf), ','.join(list(map(str, float_buf))))

    return c_str

#def np_to_uint8C_len(xs, storage_t, pad='0'):
#    if storage_t not in storage_ts:
#        print('storage_t: {} invalid. Options are {}.'.format(storage_t, storage_ts))
#
#    bit_range = 8
#    int_buf = []
#
#    if storage_t == 'col_major':
#        xs = xs.T
#
#    for x in xs:
#        for i in range(0, len(x), bit_range):
#            xi = x[i:i+bit_range]
#            xi = ''.join(map(str, xi))
#            xi = xi.ljust(bit_range, pad)
#
#            xi = int(xi, 2)
#            int_buf.append(xi)
#    return len(int_buf)

def np_to_uint8C(xs, name, storage_t, pad='0'):
    '''
    Converts a numpy array into a binary C array stored in uint8s
    '''
    if storage_t not in storage_ts:
        print('storage_t: {} invalid. Options are {}.'.format(storage_t, storage_ts))

    bit_range = 8
    int_buf = []

    if storage_t == 'col_major':
        xs = xs.T

    for x in xs:
        for i in range(0, len(x), bit_range):
            xi = x[i:i+bit_range]
            xi = ''.join(map(str, xi))
            xi = xi.ljust(bit_range, pad)

            xi = int(xi, 2)
            int_buf.append(xi)

    c_str = 'uint8_t {}[{}] = {{{}}};'.format(name, len(int_buf), ','.join(map(str, int_buf)))

    return c_str

def np_to_packed_uint8C(xs, name, storage_t, pad='0'):
    '''
    Converts a numpy array into a binary C array stored in uint8s
    '''
    if storage_t not in storage_ts:
        print('storage_t: {} invalid. Options are {}.'.format(storage_t, storage_ts))

    bit_range = 8
    int_buf = []

    if storage_t == 'col_major':
        xs = xs.T

    xs = xs.flatten()
    for i in range(0, len(xs), bit_range):
        xi = xs[i:i+bit_range]
        xi = ''.join(map(str, xi))
        xi = xi.ljust(bit_range, pad)
        xi = int(xi, 2)
        int_buf.append(xi)

    c_str = 'uint8_t {}[{}] = {{{}}};'.format(name, len(int_buf), ','.join(map(str, int_buf)))

    return c_str
