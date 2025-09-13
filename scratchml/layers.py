# import numpy as np
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

        # TODO: backprop, is it correct?
        # TODO: padding
        
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

    def optim_init(self, optim):
        self.W_opt = copy.copy(optim)
        self.b_opt = copy.copy(optim)

    def __call__(self, x):
        self.in_x = x
        x_shape = x.shape
        if len(x_shape) == 4:
            # (B, C, H, W)
            self.batch, _, x_height, x_width = x_shape
            # raise NotImplementedError("Batch forward-pass is not implemented")
        elif len(x_shape) == 3:
            # (C, H, W)
            x = np.expand_dims(x, 0)
            _, x_height, x_width = x_shape
            self.batch = 1

        if self.out_height is None:
            self.out_height = np.floor(((x_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1)
        if self.out_width is None:
            self.out_width = np.floor(((x_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1]) + 1)
        
        out = np.zeros((self.batch, self.out_channels, int(self.out_height), int(self.out_width)))

        for b in range(self.batch):
            for k in range(self.out_channels):
                for h in range(int(self.out_height)):
                    for w in range(int(self.out_width)):
                        out[b, k, h, w] = np.sum(self.in_x[b, :, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0]:self.dilation[0],
                                                                 w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]:self.dilation[1]] * self.weights[k]) + self.bias[k]
        return out 


    def backward(self, grad):
        _grad_out = np.zeros_like(self.in_x)
        _grad_weights = np.zeros_like(self.weights)
        _grad_bias = np.zeros_like(self.bias)

        for b in range(self.batch):
            for k in range(self.out_channels):
                for h in range(int(self.out_height)):
                    for w in range(int(self.out_height)):
                        _grad_weights[k] = grad[b, k, h, w] * self.in_x[b, :, h*self.stride[0]:h*self.stride[1]+self.kernel_size[0]:self.dilation[0],
                                                                              w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]:self.dilation[1]]
                        _grad_bias = grad[b, k, h, w]
                        _grad_out[b, :, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0]:self.dilation[0],
                                        w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]:self.dilation[1]] = grad[b, k, h, w] * self.weights[k]
        
        self.weights = self.W_opt.update(self.weights, _grad_weights)
        self.b = self.b_opt.update(self.bias, _grad_bias)

        return _grad_out

class AvgPooling():
    def __init__(
            self,
            kernel_size,
            stride = 1,
            padding = 0,
    ):
        # TODO: padding
        # TODO: stride
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

        self.out_height = None
        self.out_width = None

        self.weights = None
        

    def __call__(self, x):
        self.in_x = x
        x_shape = x.shape
        if len(x_shape) == 4:
            # (B, C, H, W)
            self.batch, self.channels, x_height, x_width = x_shape
        elif len(x_shape) == 3:
            # (C, H, W)
            x = np.expand_dims(x, 0)
            self.channels, x_height, x_width = x_shape
            self.batch = 1

        if self.weights is None:
            self.weights  = np.full((self.channels, self.channels, self.kernel_size[0], self.kernel_size[1]), 1/(self.kernel_size[0] * self.kernel_size[1]))
        if self.out_height is None:
            self.out_height = np.floor((x_height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        if self.out_width is None:
            self.out_width = np.floor((x_width + 2 * self.padding[1] -  self.kernel_size[1]) / self.stride[1] + 1)
        
        out = np.zeros((self.batch, self.channels, int(self.out_height), int(self.out_width)))

        for b in range(self.batch):
            for k in range(self.channels):
                for h in range(int(self.out_height)):
                    for w in range(int(self.out_width)):
                        out[b, k, h, w] = np.sum(self.in_x[b, :, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0],
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

    def backward(self, grad):
        batch, channels, height, width = self.in_x.shape
        reshaped_grad = grad.reshape((batch, channels, height, width))
        return reshaped_grad
