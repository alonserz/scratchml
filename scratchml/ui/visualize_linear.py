import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from scratchml.optimizer import Adam, RMSProp, SGD
from scratchml.utils import Sequence, to_categorical
from scratchml.layers import Linear, ReLU, Softmax, Sigmoid, Dropout
from scratchml.loss import CrossEntropy
from scratchml.utils import STDScaler
import time 

# TODO: this is demo, i guess i should make it as GUI app, not web

def plot_decision_boundary(model, reduce_dim_model, df_pca, y_train, tk_root, grid_points_amount = 100):
    # TODO: very strange behavior, accuracy is not equal to what i can see on plot
    # for example: accuracy metric score for iris dataset is 0.9(6), but sometimes
    # plot shows that some points are misclassified.


    # idea behind: we trained our model on ORIGINAL dataset with N-dims
    # then we fit ORIGINAL dataset to any model that can reduce dimension (like PCA or TSNE, etc. (AND CAN INVERSE TRANSFORM))
    # creating grid with x_min, x_max, y_min, y_max values based on reduced dim dataset (in my case Nd -> 2d (also can do 3d, just add z-axis))
    # in that case we got two sets along x and y axes, each sample from dataset represented
    # as single point with coordinates (xn, yn), where n is index of sample from dataset
    # then we're inverting set of {(x0, y0), (x1, y1), ..., (xn, yn)} to get original (N) dimension that model trained on
    # and then just forward-pass that inverted points to model and plot result

    _, ax = plt.subplots()

    x_min, x_max = df_pca[:, 0].min() - 0.2, df_pca[:, 0].max() + 0.2
    y_min, y_max = df_pca[:, 1].min() - 0.2, df_pca[:, 1].max() + 0.2

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points_amount),
                            np.linspace(y_min, y_max, grid_points_amount))

    inversed_grid = reduce_dim_model.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

    
    predicts = model(inversed_grid)
    predicts = predicts.argmax(axis = 1)
    predicts = predicts.reshape(xx.shape)

    ax.contourf(xx, yy, predicts, alpha=0.7)
    ax.scatter(df_pca[:, 0], df_pca[:, 1], c = np.argmax(y_train, axis = 1), edgecolors='k')

    

def train_model(X_train, y_train, n_epochs = 100, show_loss_while_training = False):
    model = Sequence(
        Linear(64, 512),
        Dropout(p = 0.3),
        ReLU(),
        Linear(512, 1028),
        Dropout(p = 0.3),
        ReLU(),
        Linear(1028, 10),
        Softmax(),
    )

    optim = Adam(model, lr = 4e-3)
    loss = CrossEntropy()

    def wrapper():
        model.train()
        y_pred = model(X_train)
        _loss = loss(y_train, y_pred)
        if show_loss_while_training:
            print(np.mean(_loss))
        loss_grad = loss.gradient(y_train, y_pred)
        model.backward(loss_grad)

        return model, np.mean(_loss)

    return wrapper 

if __name__ == '__main__':
    dataset = load_digits()
    X = dataset.data
    y = dataset.target
    _y = to_categorical(y.astype("int"))
    epoch = 0
    X_train, X_test, y_train, y_test = train_test_split(X, _y, test_size=0.2)

    scaler = STDScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    pca = PCA(n_components = 2, random_state = 42)
    df_pca_train = pca.fit_transform(X_train)
    df_pca_test = pca.transform(X_test)

    train_func = train_model(X_train, y_train)

    # GUI app
    root = tk.Tk()
    root.title("Linear model visualization")
    root.geometry("500x500")

    epoch_label = ttk.Label(root, text = "Epoch")
    loss_label = ttk.Label(root, text = "Loss")
    
    # embed plot to tkinter
    fig, ax = plt.subplots()
    model, loss = train_func()
    
    grid_points_amount = 100
    x_min, x_max = df_pca_train[:, 0].min() - 0.2, df_pca_train[:, 0].max() + 0.2
    y_min, y_max = df_pca_train[:, 1].min() - 0.2, df_pca_train[:, 1].max() + 0.2

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_points_amount),
                            np.linspace(y_min, y_max, grid_points_amount))

    inversed_grid = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

    model, loss = train_func()
    epoch_label.config(text = "Epoch: " + str(epoch))
    loss_label.config(text = "Loss: " + str(loss))
    predicts = model(inversed_grid)
    predicts = predicts.argmax(axis = 1)
    predicts = predicts.reshape(xx.shape)

    grid = ax.contourf(xx, yy, predicts, alpha=0.7)
    ax.scatter(df_pca_train[:, 0], df_pca_train[:, 1], c = np.argmax(y_train, axis = 1), edgecolors='k')

    def next_epoch():
        global ax, epoch 
        epoch += 1
        model, loss = train_func()
        model.eval()
        epoch_label.config(text = "Epoch: " + str(epoch))
        loss_label.config(text = "Loss: " + str(loss))
        predicts = model(inversed_grid)
        predicts = predicts.argmax(axis = 1)
        predicts = predicts.reshape(xx.shape)

        ax.clear()
        ax.contourf(xx, yy, predicts, alpha=0.7)
        ax.scatter(df_pca_train[:, 0], df_pca_train[:, 1], c = np.argmax(y_train, axis = 1), edgecolors='k')
        canvas.draw()

    def run_mil_epochs(i = 0):
        global ax, epoch 
        if i < 1_000_000: 
            epoch += 1
            model, loss = train_func()
            model.eval()
            epoch_label.config(text = "Epoch: " + str(epoch))
            loss_label.config(text = "Loss: " + str(loss))
            predicts = model(inversed_grid)
            predicts = predicts.argmax(axis = 1)
            predicts = predicts.reshape(xx.shape)

            ax.clear()
            ax.contourf(xx, yy, predicts, alpha=0.7)
            ax.scatter(df_pca_train[:, 0], df_pca_train[:, 1], c = np.argmax(y_train, axis = 1), edgecolors='k')
            canvas.draw()
            root.after(100, run_mil_epochs, i + 1)

    next_epoch_button = ttk.Button(
        root,
        text = "Next epoch",
        command = next_epoch,
    )

    mil_epochs_button = ttk.Button(
        root,
        text = "Run 1m epochs",
        command = run_mil_epochs,
    )


    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    epoch_label.pack(side = tk.TOP)
    loss_label.pack(side = tk.TOP)
    next_epoch_button.pack(side = tk.TOP)
    mil_epochs_button.pack(side = tk.TOP)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()
