import numpy as np

class Sequence():
    def __init__(self, *args):
        self.layers = args
        self.training = True
        self.optim = None

    def __call__(self, x):
        # forward pass
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = self.training
            x = layer(x)
        return x

    def __iter__(self,):
        return iter(self.layers)

    def __str__(self,):
        layers = (str(layer) for layer in self.layers)
        return '\n'.join(layers)

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self,):
        self.training = True

    def eval(self,):
        self.training = False

def to_categorical(x):
    # https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/utils/data_manipulation.py
    one_hot = np.zeros((x.shape[0], int(np.amax(x)) + 1))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

class STDScaler():
    def __init__(self,):
        self.mean = 0
        self.std = 0

    def fit(self, x):
        self.mean = np.mean(x)
        self.std = np.std(x)

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        _ = self.fit(x)
        return self.transform(x)
