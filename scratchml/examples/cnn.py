import numpy as np

from scratchml.layers import Conv, AvgPooling, ReLU, Flatten
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state

from scratchml.optimizer import Adam, RMSProp, SGD
from scratchml.utils import Sequence, to_categorical
from scratchml.layers import Linear, ReLU, Softmax
from scratchml.loss import CrossEntropy
from scratchml.utils import STDScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_ds, y_ds = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)



random_state = check_random_state(0)
permutation = random_state.permutation(X_ds.shape[0])
X_ds = X_ds[permutation]
y_ds = y_ds[permutation]

X = X_ds[:100]
y = y_ds[:100]
y = np.array(list(map(lambda x: int(x), y)))
X = X.reshape((X.shape[0], -1))
y = to_categorical(y)
X = X.reshape((100, 28, 28))

X = np.expand_dims(X, axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

scaler = STDScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequence(
    Conv(1, 6, 5),
    AvgPooling(2, stride = 2),
    ReLU(),
    Conv(6, 16, 5),
    AvgPooling(2, stride = 2),
    ReLU(),
    Conv(16, 120, 4),
    Flatten(),
    Linear(120, 80),
    ReLU(),
    Linear(80, 10),
    Softmax()
)

optim = SGD(model, lr = 3e-3, weight_decay = 3e-4)
loss = CrossEntropy()

def train(X, y, n_epochs = 200):
    for epoch in range(n_epochs):
        print(f"Epoch #{epoch}")
        y_pred = model(X_train)
        _loss = loss(y_train, y_pred)
        print(np.mean(_loss))
        print(accuracy_score(np.argmax(y_train, axis = 1), np.argmax(y_pred, axis = 1)))
        loss_grad = loss.gradient(y_train, y_pred)
        model.backward(loss_grad)

        y_pred = model(X_test)
        print(accuracy_score(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1)))

    
train(X, y, 200)
