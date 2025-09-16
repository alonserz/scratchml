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

    def __str__(self,):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features})"

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

    def __str__(self,):
        return f"{self.__class__.__name__}()"

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

    def __str__(self,):
        return f"{self.__class__.__name__}()"

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

    def __str__(self,):
        return f"{self.__class__.__name__}()"
    
    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

    def backward(self, grad):
        return grad * self.gradient(self.in_x)

class Conv():
    # https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks#layer
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            padding = 0,
            dilation = 1,
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation   

        self.out_height = None
        self.out_width = None
        self.in_x = None
        
        # weights init according to https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        k = 1/(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        k_sqrt = np.sqrt(k)
        self.weights  = np.random.uniform(-k_sqrt, k_sqrt, (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = np.random.uniform(-k_sqrt, k_sqrt, (out_channels))

        self.W_opt = None
        self.b_opt = None

        self.padded_x = None

    def optim_init(self, optim):
        self.W_opt = copy.copy(optim)
        self.b_opt = copy.copy(optim)

    def __str__(self,):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})"

    def __call__(self, x):
        self.in_x = x
        if len(self.in_x.shape) == 3:
            # (C, H, W)
            self.in_x = np.expand_dims(self.in_x, 0)

        self.batch, _, x_height, x_width = self.in_x.shape 

        if self.out_height is None:
            self.out_height = np.floor(((x_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1)
        if self.out_width is None:
            self.out_width = np.floor(((x_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1]) + 1)
        
        out = np.zeros((self.batch, self.out_channels, int(self.out_height), int(self.out_width)))
        self.padded_x = np.pad(self.in_x, [(0, 0), (0, 0), self.padding, self.padding]) 

        for b in range(self.batch):
            for k in range(self.out_channels):
                for h in range(int(self.out_height)):
                    for w in range(int(self.out_width)):
                        out[b, k, h, w] = np.sum(self.padded_x[b, :, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0]:self.dilation[0],
                                                          w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]:self.dilation[1]] * self.weights[k]) + self.bias[k]
        return out 


    def backward(self, grad):
        _grad_out = np.zeros_like(self.in_x)
        _grad_weights = np.zeros_like(self.weights)
        _grad_bias = np.zeros_like(self.bias)

        _grad_out = np.pad(_grad_out, [(0, 0), (0,0), self.padding, self.padding])

        for b in range(self.batch):
            for k in range(self.out_channels):
                for h in range(int(self.out_height)):
                    for w in range(int(self.out_height)):
                        _grad_weights[k] += grad[b, k, h, w] * self.padded_x[b, :, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0]:self.dilation[0],
                                                                              w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]:self.dilation[1]]
                        _grad_bias += grad[b, k, h, w]
                        _grad_out[b, :, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0]:self.dilation[0],
                                        w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]:self.dilation[1]] = grad[b, k, h, w] * self.weights[k]

        if sum(self.padding) > 0:
            _grad_out = _grad_out[:, :, self.padding[0]:-self.padding[1], self.padding[0]:-self.padding[1]]
            # _grad_out[b, :, :, :] = _grad_out_pad[]
            # print(f"{_grad_out_pad[b, :, self.padding[0]:-self.padding[1], self.padding[0]:-self.padding[1]]=}")
        self.weights = self.W_opt.update(self.weights, _grad_weights)
        self.bias = self.b_opt.update(self.bias, _grad_bias)

        return _grad_out

class AvgPooling():
    def __init__(
            self,
            kernel_size,
            stride = None,
            padding = 0,
    ):
        # TODO: padding
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        else:
            if isinstance(stride, int):
                self.stride = (stride, stride)
            else:
                self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.out_height = None
        self.out_width = None

        self.weights = None
        
    def __str__(self,):
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

    def __call__(self, x):
        self.in_x = x
        if len(self.in_x.shape) == 3:
            # (C, H, W)
            self.in_x = np.expand_dims(self.in_x, 0)
        self.batch, self.channels, x_height, x_width = self.in_x.shape 

        if self.weights is None:
            self.weights  = np.full((self.channels, self.channels, self.kernel_size[0], self.kernel_size[1]), 1/(self.kernel_size[0] * self.kernel_size[1]))
        if self.out_height is None:
            self.out_height = np.floor((x_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        if self.out_width is None:
            self.out_width = np.floor((x_width + 2 * self.padding[1] -  self.kernel_size[1]) / self.stride[1] + 1)
        
        out = np.zeros((self.batch, self.channels, int(self.out_height), int(self.out_width)))
        padded_x = np.pad(self.in_x, [(0, 0), (0, 0), self.padding, self.padding]) 

        for b in range(self.batch):
            for k in range(self.channels):
                for h in range(int(self.out_height)):
                    for w in range(int(self.out_width)):
                        out[b, k, h, w] = np.sum(padded_x[b, :, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
                                                                 w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]] * self.weights[k])
        return out 


    def backward(self, grad):
        _grad_out = np.zeros_like(self.in_x)

        for b in range(self.batch):
            for k in range(self.channels):
                for h in range(int(self.out_height)):
                    for w in range(int(self.out_width)):
                        _grad_out[b, :, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
                                        w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]] = grad[b, k, h, w] * self.weights[k]
        
        return _grad_out


class Flatten():
    def __init__(self,):
        self.in_x = None

    def __call__(self, x):
        self.in_x = x
        return x.reshape((x.shape[0], -1))

    def __str__(self,):
        return f"{self.__class__.__name__}()"

    def backward(self, grad):
        batch, channels, height, width = self.in_x.shape
        reshaped_grad = grad.reshape((batch, channels, height, width))
        return reshaped_grad

class Dropout():
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    def __init__(self, p = 0.5):
        self.p = p
        self.training = True
        self.mask = None

    def __str__(self,):
        return f"{self.__class__.__name__}(p={self.p})"

    def __call__(self, x):
        if self.training:
            self.mask = np.random.uniform(size = x.shape) > self.p
            return (x * self.mask) / (1 - self.p)
        return x

    def backward(self, grad):
        return (grad * self.mask) / (1 - self.p)

class BatchNorm2D():
    # https://medium.com/@ghoshanurag66/batch-normalization-math-and-implementation-fe06293f7443
    def __init__(self, n_features, eps = 1e-05, momentum = 0.1):
        self.n_features = n_features
        self.eps = eps 
        self.momentum = momentum
        # non-learnable
        self.mov_mean = np.full((self.n_features), 0)
        self.mov_var = np.full((self.n_features), 0)
        # learnable
        self.gamma = np.full((self.n_features), 1)
        self.beta = np.full((self.n_features), 0)

        self.training = True
        self.in_x = None
        self.cache = ()

        self.gamma_opt = None
        self.beta_opt = None

    def __str__(self,):
        return f"{self.__class__.__name__}(n_features={self.n_features}, eps={self.eps}, momentum={self.momentum})"
        
    def optim_init(self, optim):
        self.gamma_opt = copy.copy(optim)
        self.beta_opt = copy.copy(optim)

    def __call__(self, x):
        self.in_x = x
        if self.training:
            # calc per-feature variance and mean across mini-batch 
            _mean = np.mean(x, axis = (0, 2, 3))
            _var = np.var(x, axis = (0, 2, 3))


            self.mov_mean = self.momentum * self.mov_mean + (1 - self.momentum) * _mean
            self.mov_var = self.momentum * self.mov_var + (1 - self.momentum) * _var
            
        else:
            _mean = self.mov_mean
            _var = self.mov_var

        _mean = _mean[None, :, None, None]
        _var = _var[None, :, None, None]
        # https://numpy.org/doc/stable/user/basics.indexing.html#dimensional-indexing-tools
        x_hat = (x - _mean) / np.sqrt(_var + self.eps)
        y = self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]

        self.cache = (x_hat, _mean, _var)
        return y

    def backward(self, grad):
        x_hat, _mean, _var = self.cache

        # TODO: is this part with gamma and beta params update actually correct?
        _grad_beta = np.sum(grad, axis = 0)
        _grad_gamma = np.sum(grad * x_hat, axis = 0)

        _grad_beta = np.sum(_grad_beta, axis = (1, 2))
        _grad_gamma = np.sum(_grad_gamma, axis = (1, 2))

        self.gamma = self.gamma_opt.update(self.gamma, _grad_gamma)
        self.beta = self.beta_opt.update(self.beta, _grad_beta)

        N = grad.shape[0]

        dxhat = (grad * self.gamma[None, :, None, None])
        dmean = (1/N) * (np.sum(dxhat, axis = 0) * (-1)/np.sqrt(_var + self.eps))
        dxij = dxhat / np.sqrt(_var + self.eps)

        dvar = np.sum(dxhat * (self.in_x - _mean), axis = 0) * (-1) * np.power(_var + self.eps, -1.5) * (self.in_x - _mean) / N
        
        _grad_out = dxij + dmean + dvar

        return _grad_out

class BatchNorm1D():
    # https://medium.com/@ghoshanurag66/batch-normalization-math-and-implementation-fe06293f7443
    def __init__(self, n_features, eps = 1e-05, momentum = 0.1):
        self.n_features = n_features
        self.eps = eps 
        self.momentum = momentum
        # non-learnable
        self.mov_mean = np.full((self.n_features), 0)
        self.mov_var = np.full((self.n_features), 0)
        # learnable
        self.gamma = np.full((self.n_features), 1)
        self.beta = np.full((self.n_features), 0)

        self.training = True
        self.in_x = None
        self.cache = ()

        self.gamma_opt = None
        self.beta_opt = None

    def __str__(self,):
        return f"{self.__class__.__name__}(n_features={self.n_features}, eps={self.eps}, momentum={self.momentum})"
        
    def optim_init(self, optim):
        self.gamma_opt = copy.copy(optim)
        self.beta_opt = copy.copy(optim)

    def __call__(self, x):
        self.in_x = x
        if self.training:
            # calc per-feature variance and mean across mini-batch 
            _mean = np.mean(x, axis = 0)
            _var = np.var(x, axis = 0)

            self.mov_mean = self.momentum * self.mov_mean + (1 - self.momentum) * _mean
            self.mov_var = self.momentum * self.mov_var + (1 - self.momentum) * _var
            
        else:
            _mean = self.mov_mean
            _var = self.mov_var

        # https://numpy.org/doc/stable/user/basics.indexing.html#dimensional-indexing-tools
        x_hat = (x - _mean) / np.sqrt(_var + self.eps)
        y = self.gamma * x_hat + self.beta

        self.cache = (x_hat, _mean, _var)
        return y

    def backward(self, grad):
        x_hat, _mean, _var = self.cache

        # TODO: is this part with gamma and beta params update actually correct?
        _grad_beta = np.sum(grad, axis = 0)
        _grad_gamma = np.sum(grad * x_hat, axis = 0)

        self.gamma = self.gamma_opt.update(self.gamma, _grad_gamma)
        self.beta = self.beta_opt.update(self.beta, _grad_beta)

        N = grad.shape[0]

        dxhat = (grad * self.gamma)
        dmean = (1/N) * (np.sum(dxhat, axis = 0) * (-1)/np.sqrt(_var + self.eps))
        dxij = dxhat / np.sqrt(_var + self.eps)

        dvar = np.sum(dxhat * (self.in_x - _mean), axis = 0) * (-1) * np.power(_var + self.eps, -1.5) * (self.in_x - _mean) / N
        
        _grad_out = dxij + dmean + dvar

        return _grad_out
