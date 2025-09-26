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

def split_batch_train_test(
        X,
        y,
        batch_size = 1,
        test_size = 0.3,
        drop_last = True,
        shuffle = True,
):
    assert batch_size > 0, "Batch Size can't be less than 0"
    assert test_size >= 0 and test_size <= 1, "Value of test size must be in range [0, 1]"

    # Reshape dataset to batches with dropping last elements
    drop_last_size = X.shape[0] - (X.shape[0] % batch_size)
    X = X[:drop_last_size]
    y = y[:drop_last_size]

    X = X.reshape(X.shape[0] // batch_size, batch_size, *X.shape[1:])
    y = y.reshape(y.shape[0] // batch_size, batch_size, *y.shape[1:])

    # shuffle
    if shuffle:
        perm = np.random.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]


    train_size = int((1 - test_size) * X.shape[0])
    X_train = X[:train_size]
    y_train = y[:train_size]
    assert X_train.shape[0] == y_train.shape[0]
    X_test = X[train_size:]
    y_test = y[train_size:]
    assert X_test.shape[0] == y_test.shape[0]


    return X_train, X_test, y_train, y_test
