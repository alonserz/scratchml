import numpy as np
from scratchml.layers import Conv, AvgPooling, ReLU, Flatten, Dropout, BatchNorm2D
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


n_samples = 200

X = X_ds[:n_samples]
y = y_ds[:n_samples]
y = np.array(list(map(lambda x: int(x), y)))
y = to_categorical(y)
X = X.reshape((n_samples, 28, 28))

X = np.expand_dims(np.array(X), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

scaler = STDScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequence(
    Conv(1, 6, 5, padding = 2), # (1x28x28) -> (1x32x32)
    AvgPooling(2),
    BatchNorm2D(6),
    Dropout(p = 0.1),
    ReLU(),
    Conv(6, 16, 5),
    AvgPooling(2),
    BatchNorm2D(16),
    Dropout(p = 0.1),
    ReLU(),
    Conv(16, 120, 5),
    BatchNorm2D(120),
    Flatten(),
    Linear(120, 80),
    Dropout(p = 0.1),
    ReLU(),
    Linear(80, 10),
    Softmax()
)

optim = Adam(model, lr = 3e-3)
loss = CrossEntropy()

def train(X, y, n_epochs = 200):
    model.train()
    for epoch in range(n_epochs):
        print(f"Epoch #{epoch}")
        y_pred = model(X_train)
        _loss = loss(y_train, y_pred)
        print("Train Loss:", np.mean(_loss))
        print("Train accuracy: ", accuracy_score(np.argmax(y_train, axis = 1), np.argmax(y_pred, axis = 1)))
        loss_grad = loss.gradient(y_train, y_pred)
        model.backward(loss_grad)

    model.eval()
    y_pred = model(X_test)
    print("Test accuracy: ", accuracy_score(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1)))

    
train(X, y, 10)
