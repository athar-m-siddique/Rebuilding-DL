# ======================================================================
# IMPORTS ==============================================================
# ======================================================================

import matplotlib.pyplot as plt
import json
import numpy as np





# ======================================================================
# PREPARING DATA =======================================================
# ======================================================================

def load_data(nb, ne):

    points = []
    for i in range(nb,ne+1):
        with open(f"batches/{i}.json", 'r') as file:
            points.extend(json.load(file))

    points = np.array(points)

    X = points[:,:2]
    y = points[:,2].reshape(-1,1)

    print(X.shape)
    print(y.shape)

    return X,y

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X-mean)/std
    return X, mean, std





# ======================================================================
# ALGORITHM ============================================================
# ======================================================================

def initialisation(X):
    W = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1)
    return (W,b)

def model(X,W,b):
    Z = X.dot(W)+b
    A = 1/(1+np.exp(-Z))
    return A

def log_loss(A,y):
    m = len(y)
    eps = 1e-15
    A = np.clip(A, eps, 1-eps)
    return (-1/m)*np.sum(y*np.log(A)+(1-y)*np.log(1-A))

def gradients(A,X,y):
    dW = (1/len(y))*np.dot(X.T, (A-y))
    db = (1/len(y))*np.sum(A-y)
    return dW,db

def update(W,b,dW,db,lr):
    W -= lr*dW
    b -= lr*db
    return W,b

def train_neuron(X,y, lr, n_iter):
    Loss = []
    W,b = initialisation(X)
    for i in range(n_iter):
        A = model(X,W,b)
        Loss.append(log_loss(A,y))
        dW,db = gradients(A,X,y)
        W,b = update(W,b,dW,db, lr)
    return W,b,Loss





# ======================================================================
# MAIN =================================================================
# ======================================================================

if __name__ == "__main__":

    # loading data
    X_raw, y = load_data(1,8)
    X_norm, mean, std = normalize(X_raw)

    #training the neuron
    W, b, Loss = train_neuron(X_norm, y, lr=0.5, n_iter=20000)

    #getting the equation of the line in real space
    w1, w2 = W[0, 0], W[1, 0]
    b_val = b[0]
    pente = -(w1 / std[0]) * (std[1] / w2)
    ordonnee = mean[1] - pente * mean[0] - (b_val * std[1] / w2)

    print("\n--- EQUATION ---")
    print(f"Length = {pente:.4f} * red proportion + {ordonnee:.4f}\n")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_raw[:, 0], X_raw[:, 1], c=y[:, 0], cmap="bwr", alpha=0.7)

    x1_line = np.linspace(X_raw[:, 0].min(), X_raw[:, 0].max(), 100)
    x2_line = pente * x1_line + ordonnee

    plt.plot(
        x1_line,
        x2_line,
        color="green",
        linewidth=2,
        label=f"y = {pente:.2f}x + {ordonnee:.2f}",
    )
    plt.xlabel("Red Proportion")
    plt.ylabel("Length (cm)")
    plt.title("Decision Boundary (Real Data)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(Loss)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Convergence de la Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    X_test, y_test = load_data(9,10)
    X_test_norm, mean_test, std_test = normalize(X_test)

    A_test = model(X_test_norm, W, b)
    y_pred = (A_test > 0.5).astype(int)
    comp = np.hstack((y_pred, y_test))
    print("reussites : ", np.sum(y_pred == y_test), " / ", len(y_test))
    print(comp)

