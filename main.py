# ============================================================
# IMPORTS ====================================================
# ============================================================

import numpy as np
import json
import matplotlib.pyplot as plt





# ============================================================
# PREPARING DATA =============================================
# ============================================================

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    return X,y

def normalize(X, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    X = (X - mean) / std
    return X, mean, std





# ============================================================
# CLASS : LAYER ==============================================
# ============================================================

class Layer :
    def __init__(self) :
        pass





# ============================================================
# CLASS : DENSELAYER =========================================
# ============================================================

class DenseLayer (Layer) :

    def __init__(self, n_neurons, n_entries):
        super().__init__()

        self.W = np.random.randn(n_entries, n_neurons) * np.sqrt(2/n_entries)
        self.b = np.zeros((1, n_neurons))

    def forward(self, X:np.array):
        self.X = X
        self.Z =  self.X.dot(self.W)+self.b
        return self.Z

    def backward(self, dA, y, lr):
        m = self.X.shape[0]
        self.dW = self.X.T.dot(dA)
        self.db = np.sum(dA, axis=0, keepdims=True)

        dZ = dA.dot(self.W.T)
        self.W -= lr * self.dW
        self.b -= lr * self.db

        self.X = None
        self.Z = None
        return dZ

    def get_n_neurons(self):
        return self.W.shape[1]

    def get_n_entries(self):
        return self.W.shape[0]





# ============================================================
# CLASS : ACTIVATION =========================================
# ============================================================

class Activation (Layer) :

    def __init__(self, type):
        super().__init__()
        self.type=type

    def forward(self, Z: np.array):
        self.Z = Z
        match self.type :
            case "reLU" :
                self.A = np.maximum(0, Z)
            case "leakyreLU" :
                self.A = np.where(Z > 0, Z, 0.01 * Z)
            case "sigmoid" :
                clipped_input = np.clip(Z, -500, 500)
                self.A = 1 / (1 + np.exp(-clipped_input))
            case _ :
                self.A = Z

        return self.A


    def backward(self, dA, y, lr):
        match self.type :
            case "reLU" :
                dZ = dA*(self.Z>0).astype(float)
            case "leakyreLU" :
                dZ = dA*(np.where(self.Z > 0, 1, 0.01))
            case "sigmoid" :
                s = self.A
                dZ = dA*s*(1-s)
            case _ :
                dZ = dA

        return dZ





# ============================================================
# CLASS : NETWORK ============================================
# ============================================================
        
class Network :

    def __init__(self):

        self.layers : list[Layer] = []
        self.loss : list[float] = []
        self.inputs : int = 0
        self.outputs : int = 0

    def add(self, layer:Layer):
        self.layers.append(layer)        

    def forward (self, X) :
        input=X
        for L in self.layers :
            input=L.forward(input)
        return input

    def backward (self, y_pred, y, lr, weight) :
        grad = self.dlog_loss(y_pred, y, weight)
        for L in reversed(self.layers):
            grad = L.backward(grad, y, lr)


    def log_loss(self, y, weight) :
        y_pred = self.layers[-1].A
        m = len(y)
        l = (-1/m) * np.sum(weight*y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
        self.loss.append(l)
        return l

    def dlog_loss(self, y_pred2, y, weight) :
        y_pred = self.layers[-1].A
        m = len(y)
        
        numerator = weight * y * (1 - y_pred) - (1 - y) * y_pred
        denominator = y_pred * (1 - y_pred)

        dl = (-1/m) * numerator / denominator
        
        return dl

    def get_coefs(self):
        coefs = []
        for L in self.layers:
            if isinstance(L, DenseLayer):
                coefs.append((L.W, L.b))
        return coefs
        

    def train(self, X, y, epochs, lr, weight) :

        self.loss.clear()

        for e in range(epochs):
            y_pred = self.forward(X)
            self.log_loss(y, weight)
            self.backward(y_pred, y, lr, weight)

        return self.loss





# ============================================================
# MAIN FUNCTION ==============================================
# ============================================================

def main ():

    # Load and normalize the training data
    X,y = load_data("batches/train_data.json")
    X, mean, std = normalize(X)
    print(X.shape, y.shape)


    # building the neural network
    layers = [X.shape[1], 32, 16, 8, 8, 4, 1]
    activations = ["leakyreLU", "leakyreLU", "leakyreLU", "leakyreLU", "leakyreLU", "sigmoid"]

    nw=Network()

    for i in range(len(layers) - 1):
        nw.add(DenseLayer(n_entries=layers[i], n_neurons=layers[i + 1]))
        nw.add(Activation(type=activations[i]))


    # Training the neural network
    nw.train(X,y,epochs =3000, lr=0.05, weight=2.0)


    # Plotting the loss curve
    plt.figure(1)
    plt.plot(nw.loss)
    plt.show()

    X_test, y_test = load_data("batches/test_data.json")
    X_test, _, _ = normalize(X_test, mean, std)
    y_pred = (nw.forward(X_test) >= 0.5).astype(int)
    y_comp = np.hstack((y_test, y_pred))
    print(y_comp)


    # Plotting the decision boundary
    plt.figure(2)
    res = 200

    x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    
    grid_x1, grid_x2 = np.meshgrid(
        np.linspace(x1_min, x1_max, res),
        np.linspace(x2_min, x2_max, res)
    )

    grid_data = np.c_[grid_x1.ravel(), grid_x2.ravel()]
    grid_probs = nw.forward(grid_data).reshape(res, res)

    y_flat = y.ravel()

    plt.scatter(X[y_flat == 0, 0], X[y_flat == 0, 1], color='blue', label='Class 0', alpha=0.5)
    plt.scatter(X[y_flat == 1, 0], X[y_flat == 1, 1], color='red', label='Class 1', alpha=0.5)

    plt.contour(grid_x1, grid_x2, grid_probs, levels=[0.5], colors='green', linewidths=2.5)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundary')
    plt.legend()
    plt.show()


    # calculating metrics
    TP = np.sum((y_pred == 1) & (y_test == 1))
    TN = np.sum((y_pred == 0) & (y_test == 0))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))
    
    total = len(y_test)
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    rappel = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"(TP) : {TP}")
    print(f"(TN) : {TN}")
    print(f"(FP) : {FP}")
    print(f"(FN) : {FN}")
    print("----------------------------")
    print(f"(Accuracy) : {accuracy * 100:.2f}%")
    print(f"(Precision)        : {precision * 100:.2f}%")
    print(f"(Recall)              : {rappel * 100:.2f}%")

if __name__ == "__main__":
    main()