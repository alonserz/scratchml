import numpy as np
import tkinter as tk
import types
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import matplotlib.pyplot as plt
from scratchml.layers import Conv, AvgPooling, ReLU, Flatten, Dropout, BatchNorm2D, MaxPooling
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from scratchml.optimizer import Adam, RMSProp, SGD
from scratchml.utils import Sequence, to_categorical
from scratchml.layers import Linear, ReLU, Softmax
from scratchml.loss import CrossEntropy
from scratchml.utils import STDScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Stop():
    def __call__(self, x):
        return x
    def backward(self, grad):
        return grad

X_ds, y_ds = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

n_samples = 3000

X = X_ds[:n_samples]
y = y_ds[:n_samples]
y = np.array(list(map(lambda x: int(x), y)))
y = to_categorical(y)
X = X.reshape((n_samples, 28, 28))

X = np.expand_dims(np.array(X), axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

n_limit = 200

X_train, X_test, y_train, y_test = X_train[:n_limit], X_test[:n_limit], y_train[:n_limit], y_test[:n_limit]
scaler = STDScaler()
#X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
model = Sequence(
        Conv(1, 6, 5, padding = 2), # (1x28x28) -> (1x32x32)
        AvgPooling(2),
        Stop(), # Put this class after layer that you want to visualize 
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
    for epoch in range(n_epochs):
        model.train()
        print("Epoch:", epoch)
        y_pred = model(X_train)
        _loss = loss(y_train, y_pred)
        print("Train Loss:", np.mean(_loss))
        print("Train accuracy: ", accuracy_score(np.argmax(y_train, axis = 1), np.argmax(y_pred, axis = 1)))
        loss_grad = loss.gradient(y_train, y_pred)
        model.backward(loss_grad)
    return model, np.mean(_loss)
    
if __name__ == '__main__':
    rows, cols = 6, 6
    fig, ax = plt.subplots(rows, cols)
    root = tk.Tk()
    root.title("Linear model visualization")
    root.geometry("500x500")
    epoch_label = ttk.Label(root, text = "Epoch")
    loss_label = ttk.Label(root, text = "Loss")

    model, loss = train(X_test, y_test, n_epochs = 1)

    def first_conv_break(self, x):
        for layer in self.layers:
            if isinstance(layer, Stop):
                break
            x = layer(x)
        return x

    model.first_conv_break = types.MethodType(first_conv_break, model)
    first_out = model.first_conv_break(X_train[113])
    c_ = first_out.shape[1]

    for i in range(cols):
        ax[0, i].imshow(X_train[0][0], cmap='gray')
        ax[0, i].set_title("Original Image")
    curr_filter = 0

    for i, j in ((x, y) for x in range(1, rows - 1) for y in range(cols)):
        if curr_filter == c_:
            break
        ax[i, j].imshow(first_out[0][curr_filter], cmap='gray')
        curr_filter += 1

    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    epoch_label.pack(side = tk.TOP)
    loss_label.pack(side = tk.TOP)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()
