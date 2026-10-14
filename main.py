import numpy as np
import json
import matplotlib.pyplot as plt
import os
import math
import copy

def load_data(filename="batches/train_data.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    X = np.array(data, dtype=np.float32)[:,:2]
    
    y = np.array(data, dtype=np.float32)[:,2].reshape(-1,1)
    
    return X, y


def init(c):
    W = []
    b = []

    for i in range(1, len(c)):
        W.append(np.random.randn(c[i-1], c[i])*np.sqrt(2.0/c[i-1]))
        b.append(np.zeros((1,c[i])))

    return W,b

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def leaky_relu(Z, alpha=0.2):
    return np.where(Z > 0, Z, Z * alpha)

def dleaky_relu(Z, alpha=0.2):
    return np.where(Z > 0, 1.0, alpha)

def modelG(X,W,b):

    Z = []
    A = [X]

    for i in range(len(W)):
        Z.append(A[i].dot(W[i])+b[i])
        A.append(Z[i] if i == len(W)-1 else leaky_relu(Z[i]))

    A.pop(0)

    return Z,A

def modelD(X,W,b):

    Z = []
    A = [X]

    for i in range(len(W)):
        Z.append(A[i].dot(W[i])+b[i])
        A.append(sigmoid(Z[i]) if i == len(W)-1 else leaky_relu(Z[i]))

    A.pop(0)

    return Z,A

def log_loss(A,y):
    m = len(y)
    eps = 1e-15
    A = np.clip(A[-1],eps,1-eps)
    return (-1/m)*np.sum(y*np.log(A) + (1-y)*np.log(1-A))


def back_propagationD(A,X,y,Z,W):

    m = len(y)
    L = len(A)
    dW, db = [None] * L, [None] * L

    dZ = (A[-1] - y) / m

    for i in reversed(range(L)):
        A_prev = A[i-1] if i > 0 else X
        dW[i] = np.dot(A_prev.T, dZ)
        db[i] = np.sum(dZ, axis=0, keepdims=True)

        if i > 0:
            dA = np.dot(dZ, W[i].T)
            dZ = dA * dleaky_relu(Z[i-1])

    dX = np.dot(dZ, W[0].T)
    return dW, db, dX

def back_propagationG(AG, z, dX_fake, ZG, WG):
    L = len(AG)
    dW, db = [None] * L, [None] * L

    dZ = dX_fake


    for i in reversed(range(L)):
        A_prev = AG[i-1] if i > 0 else z
        dW[i] = np.dot(A_prev.T, dZ)
        db[i] = np.sum(dZ, axis=0, keepdims=True)

        if i > 0:
            dA = np.dot(dZ, WG[i].T)
            dZ = dA * dleaky_relu(ZG[i-1])

    return dW, db

def update(W,b,dW,db,lr):
    for i in range(len(W)):
        W[i]-=lr*dW[i]
        b[i]-=lr*db[i]
    return W,b


def train_network(
    X_true, y_true, couchesG, couchesD, epochs_to_save, lr, weight, batchsize
):
    WG, bG = init(couchesG)
    WD, bD = init(couchesD)

    lossG = []
    lossD = []

    history_G = {}

    epochs_to_save = sorted(list(set(epochs_to_save)))
    max_epochs = max(epochs_to_save)

    for i in range(1, max_epochs + 1):
        half_batch = batchsize // 2

        # train D
        idx = np.random.randint(0, X_true.shape[0], half_batch)
        X_real = X_true[idx]
        y_real = y_true[idx]

        z = np.random.randn(half_batch, couchesG[0])
        _, AG = modelG(z, WG, bG)
        X_fake = AG[-1]
        y_fake = np.zeros((half_batch, 1))

        X = np.vstack((X_real, X_fake))
        y = np.vstack((y_real, y_fake))

        ZD, AD = modelD(X, WD, bD)
        lossD.append(log_loss(AD, y))
        dWD, dbD, _ = back_propagationD(AD, X, y, ZD, WD)
        WD, bD = update(WD, bD, dWD, dbD, lr)

        # train G
        z = np.random.randn(half_batch, couchesG[0])
        ZG, AG = modelG(z, WG, bG)
        X_fake = AG[-1]

        y_trick = np.ones((half_batch, 1))
        ZD, AD = modelD(X_fake, WD, bD)
        lossG.append(log_loss(AD, y_trick))

        _, _, dX_fake = back_propagationD(AD, X_fake, y_trick, ZD, WD)
        dWG, dbG = back_propagationG(AG, z, dX_fake, ZG, WG)
        WG, bG = update(WG, bG, dWG, dbG, lr)

        if i in epochs_to_save:
            history_G[i] = (copy.deepcopy(WG), copy.deepcopy(bG))

    return history_G, lossG, lossD


if __name__ == "__main__":
    couchesG = [2, 16, 16, 2]
    couchesD = [2, 16, 8, 1]

    X_true, y_true = load_data()

    epochs_to_save = [1000, 5000, 8000, 10000, 15000, 20000]

    print(f"Entraînement en une passe jusqu'à {max(epochs_to_save)} epochs...")
    history_G, lossG, lossD = train_network(
        X_true,
        y_true,
        couchesG,
        couchesD,
        epochs_to_save=epochs_to_save,
        lr=0.01,
        weight=2,
        batchsize=500,
    )

    num_tests = len(epochs_to_save)
    cols = 3
    rows = math.ceil(num_tests / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.ravel() if num_tests > 1 else [axes]

    z_test = np.random.randn(500, couchesG[0])

    for idx, ep in enumerate(epochs_to_save):
        WG_saved, bG_saved = history_G[ep]
        _, A_test = modelG(z_test, WG_saved, bG_saved)
        coord = A_test[-1]

        ax = axes_flat[idx]
        ax.scatter(
            X_true[:, 0], X_true[:, 1], alpha=0.2, color="blue", label="Réelles"
        )
        ax.scatter(
            coord[:, 0], coord[:, 1], color="red", alpha=0.7, label="Générées"
        )
        ax.axis("equal")
        ax.set_title(f"Epoch : {ep}")
        ax.legend()

    for j in range(num_tests, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.show()