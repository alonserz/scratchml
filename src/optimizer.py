import numpy as np

class RMSProp():
    def __init__(
        self,
        lr = 1e-3,
        alpha = 0.99,
        eps = 1e-08,
        momentum = 0,
        weight_decay = 0,
    ):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.vt = None
        self.weight_decay = weight_decay

    def update(self, weights, grad):
        # If not initialized
        if self.vt is None:
            self.vt = np.zeros(np.shape(grad))
        if self.weight_decay != 0:
            grad = grad + self.weight_decay * w

        self.vt = self.alpha * self.vt + (1 - self.alpha) * np.power(grad, 2) 
        return wweights - self.lr * grad / (np.sqrt(self.vt) + self.eps)

class SGD():
    def __init__(
            self,
            model,
            lr = 1e-3,
            weight_decay = 0,
            dampening = 0,
            nesterov = False,
            momentum = 0,
            maximize = False,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.bt = 0
        self.momentum = momentum
        self.maximize = maximize

        for layer in model:
            if 'optim_init' not in dir(layer):
                continue
            layer.optim_init(self)

    def update(self, weights, grad):
        if self.maximize:
            grad = -grad

        if self.weight_decay != 0:
            grad = grad + self.weight_decay * weights

        if self.momentum != 0:
            self.bt = self.momentum * self.bt + (1 - self.dampening) * grad
            if self.nesterov:
                grad = grad + self.momentum * self.bt
            else:
                grad = self.bt

        return weights - self.lr * grad 

class Adam():
    def __init__(
            self,
            model,
            lr = 1e-3,
            betas = (0.9, 0.999),
            eps = 1e-08,
            weight_decay = 0,
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay 

        self.m = 0
        self.v = 0
        self.v_max = 0

        for layer in model:
            if 'optim_init' not in dir(layer):
                continue
            layer.optim_init(self)

    def update(self, weights, grad):
        if self.weight_decay != 0:
            grad = grad + self.weight_decay * weights

        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grad
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * np.power(grad, 2)
        m_hat = self.m / (1 - self.betas[0])

        v_hat = self.v / (1 - self.betas[1])

        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
