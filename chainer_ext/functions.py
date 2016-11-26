from chainer import function,cuda
import chainer.functions as F
class Entropy(function.Function):
    def forward(self, x):
        xp = cuda.get_array_module(*x)
        y = x[0] * xp.log(x[0]+1e-9)
        return -xp.sum(y,1),
        
    def backward(self, x, gy):
        return gy,
    
def entropy(x):
    return Entropy()(x)