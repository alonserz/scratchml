import numpy as np
import math
import copy

class Linear():
    # https://cs231n.stanford.edu/handouts/linear-backprop.pdf
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.in_x = None
        self.weights = None
        self.W_opt = None
        self.b_opt = None

        self.b = np.zeros((1, self.out_features))
        
        limit = 1 / math.sqrt(self.in_features)
        self.weights  = np.random.uniform(-limit, limit, (self.in_features, self.out_features))

    def __call__(self, x):
        # forward pass
        self.in_x = x
        return x.dot(self.weights) + self.b

    def optim_init(self, optim):
        self.W_opt = copy.copy(optim)
        self.b_opt = copy.copy(optim)

    def backward(self, grad):
        W = self.weights

        _grad = self.in_x.T.dot(grad)
        grad_b = np.sum(grad, axis=0, keepdims=True)

        self.weights = self.W_opt.update(self.weights, _grad)
        self.b = self.b_opt.update(self.b, grad_b)


        _grad_out = grad.dot(W.T)
        return _grad_out


class Softmax():
    def __init__(self,):
        self._in_x = None

    def __call__(self, x):
        self.in_x = x
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

    def backward(self, grad):
        return grad * self.gradient(self.in_x)

class ReLU():
    def __init__(self,):
        self.in_x = None

    def __call__(self, x):
        # forward pass
        self.in_x = x
        return np.where(x >= 0, x, 0) 

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)  

    def backward(self, grad):
        return grad * self.gradient(self.in_x)

class Sigmoid():
    def __init__(self,):
        self.in_x = None

    def __call__(self, x):
        self.in_x = x
        return 1/(1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

    def backward(self, grad):
        return grad * self.gradient(self.in_x)
