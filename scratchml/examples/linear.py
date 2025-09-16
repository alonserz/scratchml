import numpy as np
import math
import copy

from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split

from scratchml.optimizer import Adam, RMSProp, SGD
from scratchml.utils import Sequence, to_categorical
from scratchml.layers import Linear, ReLU, Softmax, Dropout, BatchNorm1D
from scratchml.loss import CrossEntropy
from scratchml.utils import STDScaler
from sklearn.metrics import accuracy_score

dataset = load_digits()
X = dataset.data
y = dataset.target
_y = to_categorical(y.astype("int"))

X_train, X_test, y_train, y_test = train_test_split(X, _y, test_size=0.2, random_state =42)

scaler = STDScaler()
X_train = scaler.fit_transform(X_train)


model = Sequence(
    Linear(64, 12),
    BatchNorm1D(12),
    Dropout(p = 0.2),
    ReLU(),
    Linear(12, 12),
    BatchNorm1D(12),
    Dropout(p = 0.2),
    ReLU(),
    Linear(12, 10),
    Softmax(),
)

optim = Adam(model, lr = 1e-3)
loss = CrossEntropy()

for epoch in range(200):
    model.train()
    y_pred = model(X_train)
    _loss = loss(y_train, y_pred)
    print(np.mean(_loss))
    loss_grad = loss.gradient(y_train, y_pred)
    model.backward(loss_grad)
    print("Train accuracy: ", accuracy_score(np.argmax(y_train, axis = 1), np.argmax(y_pred, axis = 1)))

    model.eval()
    y_pred = model(X_test)
    print("Test accuracy: ", accuracy_score(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1)))
