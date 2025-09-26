import numpy as np
import math
import copy

from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split

from scratchml.optimizer import Adam, RMSProp, SGD
from scratchml.utils import Sequence, to_categorical
from scratchml.layers import Linear, ReLU, Softmax, Dropout, BatchNorm1D
from scratchml.loss import CrossEntropy
from scratchml.utils import STDScaler, split_batch_train_test
from sklearn.metrics import accuracy_score

dataset = load_digits()
X = dataset.data
y = dataset.target
_y = to_categorical(y.astype("int"))

X_train, X_test, y_train, y_test = split_batch_train_test(X, _y, test_size=0.2, batch_size = 8)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

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

for epoch in range(20):
    model.train()
    train_loss = 0
    for batch in range(X_train.shape[0]):
        y_pred = model(X_train[batch])
        train_loss += np.mean(loss(y_train[batch], y_pred))
        loss_grad = loss.gradient(y_train[batch], y_pred)
        model.backward(loss_grad)

    print("Train Loss: ", train_loss / X_train.shape[0])

    model.eval()
    test_loss = 0
    for batch in range(X_test.shape[0]):
        y_pred = model(X_test[batch])
        test_loss += np.mean(loss(y_test[batch], y_pred))
    print("Test Loss: ", test_loss / X_test.shape[0])

train_accuracy = 0
for batch in range(X_train.shape[0]):
    y_pred = model(X_train[batch])
    train_accuracy += accuracy_score(np.argmax(y_train[batch], axis = 1), np.argmax(y_pred, axis = 1)) * 1/X_train.shape[0]
print("Train accuracy", train_accuracy)


test_accuracy = 0
for batch in range(X_test.shape[0]):
    y_pred = model(X_test[batch])
    test_accuracy += accuracy_score(np.argmax(y_test[batch], axis = 1), np.argmax(y_pred, axis = 1)) * 1/X_test.shape[0]
print("Test accuracy", test_accuracy)
