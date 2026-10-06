# ======================================================================
# IMPORTS ==============================================================
# ======================================================================

import numpy as np
import json
import matplotlib.pyplot as plt





# ======================================================================
# PREPARING DATA========================================================
# ======================================================================

def import_data(mode="train"):

    filename = "batches/" + mode + "_data.json"
    points = []
    X = np.array([])
    with open(filename, 'r') as file:
        points.extend(json.load(file))
    points=np.array(points)

    X = points[:,:-1]
    y = points[:,-1].reshape(-1,1)

    return X,y



def normalize(X, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    X = (X - mean) / std
    return X, mean, std



def plot_contour_2d(X, y, W, b, resolution=200):
    
    x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    
    grid_x1, grid_x2 = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution),
        np.linspace(x2_min, x2_max, resolution)
    )
    
    grid_data = np.c_[grid_x1.ravel(), grid_x2.ravel()]

    _, A_grid = model(grid_data, W, b)
    grid_probs = A_grid[-1].reshape(resolution, resolution)

    plt.figure(figsize=(8, 6))
    
    y_flat = y.ravel()
    plt.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], c='blue', alpha=0.5, label='Classe 0')
    plt.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], c='red', alpha=0.8, label='Classe 1')
    
    plt.contour(grid_x1, grid_x2, grid_probs, levels=[0.5], colors='#00FF00', linewidths=2.5)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()





# ======================================================================
# ALGORITHM ============================================================
# ======================================================================



def initialisation(X, layers):

    l = layers.copy()
    l.insert(0, X.shape[1])

    W = []
    b = []

    for i in range(1, len(l)):
        W.append(np.random.randn(l[i-1], l[i]) * np.sqrt(2.0 / l[i-1]))
        b.append(np.zeros((1, l[i])))
    
    return W, b



def reLU(Z):
    return np.maximum(0,Z)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def dreLU(Z):
    return (Z > 0).astype(float)

def dsigmoid(Z):
    s = sigmoid(Z)
    return s*(1-s)



def log_loss(A,y):
    m = len(y)
    eps = 1e-15
    A = np.clip(A[-1], eps, 1-eps)
    return (-1/m)*np.sum(y*np.log(A)+(1-y)*np.log(1-A))

def dlog_loss(A,y):
    m = len(y)
    eps = 1e-15
    A = np.clip(A[-1], eps, 1-eps)
    return (-1/m)*np.sum((y/A) * ((y-1)/(1-A)))



def model(X,W,b):

    Z = []
    A = [X]

    for i in range(len(W)):
        Z.append(A[i].dot(W[i])+b[i])
        A.append(sigmoid(Z[i]) if i==len(W)-1 else reLU(Z[i]))

    A.pop(0)

    return Z, A



def back_propagation (A, X, y, Z, W, weight):

    m = len(y)
    L = len(A)

    dW = [None]*L
    db = [None]*L

    dA = None
    dZ=0

    for i in reversed(range(L)):

        dZ = A[i]-y if i==L-1 else dA*dreLU(Z[i])
        if i==L-1:
            dZ[y==1]*=weight
            
        A_prev=A[i-1] if i>0 else X

        dW[i] = (1/m)*np.dot(A_prev.T, dZ)
        db[i] = (1/m)*np.sum(dZ, axis=0, keepdims=True)

        dA = np.dot(dZ, W[i].T)

    return dW, db



def update(W,b,dW,db,lr):
    for i in range(len(W)):
        W[i] -= lr*dW[i]
        b[i] -= lr*db[i]
    return W,b



def train_network (X, y, layers, n_iter, lr, weight):
    loss = []
    W,b = initialisation(X,layers)
    for n in range(n_iter):
        Z,A = model(X,W,b)
        loss.append(log_loss(A,y))
        dW,db = back_propagation(A,X,y,Z,W, weight)
        update(W,b,dW,db,lr)

    return W,b,loss





# ======================================================================
# MAIN =================================================================
# ======================================================================

if __name__ == "__main__":


    # exploring different architectures
    l1 = [1]
    l2 = [1, 1, 1, 1, 1]
    l3 = [4, 1]
    l4 = [8, 4, 1]
    l5 = [8, 4, 2, 1]
    l6 = [32, 8, 2, 1]
    l7 = [32, 16, 8, 1]
    l8 = [8, 16, 32, 16, 8, 1]
    l9 = [512, 128, 32, 8, 1]

    l10= [7, 1]

    layers=l10

    #import data & normalize X
    X,y = import_data("train")
    X, mean, std = normalize(X)
    print(X.shape, y.shape)



    # initialization neuron network
    W,b = initialisation(X, layers)
    for i in range(len(W)):
        print(f"W{i+1} shape: {W[i].shape}")
        print(f"b{i+1} shape: {b[i].shape}")

    #testing model
    Z, A = model(X,W,b)
    for i in range(len(A)):
            print(f"A{i+1} shape: {A[i].shape}")

    #testing backpropag
    dW, db = back_propagation(A,X,y,Z,W, weight=1.0)
    for i in range(len(dW)):
        print(f"dW{i+1} shape: {dW[i].shape}")
        print(f"db{i+1} shape: {db[i].shape}")

    #testing training
    W,b,loss = train_network(X,y, layers, n_iter=1500, lr=0.03, weight=3.0)

    plt.plot(loss)


    #ploting
    plot_contour_2d(X, y, W, b)

    # Evaluating performances on test dataset
    X_test, y_test = import_data("test")
    X_test, _,_ = normalize(X_test, mean, std)

    Z_test, A_test = model(X_test, W, b)
    y_pred = (A_test[-1] >= 0.5 ).astype(int)

    # calculating confusion matrix and metrics
    TP = np.sum((y_pred == 1) & (y_test == 1))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))

    total = len(y_test)
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    print("\n--- CONFUSION MATRIX ---")
    print(f"(TP) : {TP}")
    print(f"(TN) : {TN}")
    print(f"(FP) : {FP}")
    print(f"(FN) : {FN}")
    print("----------------------------")
    print(f"(Accuracy) : {accuracy * 100:.2f}%")
    print(f"(Precision): {precision * 100:.2f}%")
    print(f"(Recall)   : {recall * 100:.2f}%")
    print("----------------------------")
    print(f"(F1-Score) : {(2 * precision * recall) / (precision + recall) * 100:.2f}%")


