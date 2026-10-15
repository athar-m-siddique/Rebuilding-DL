import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATA_DIR = "batches"
LR = 0.1
WEIGHT = 1
EPOCHS = 5000
LAYERS = [32, 16, 8, 1]

torch.manual_seed(0)


def load_data(mode):
    with open(f"{DATA_DIR}/{mode}_data.json", "r") as f:
        data = np.array(json.load(f))
    return data[:, :-1], data[:, -1].reshape(-1, 1)


def normalize(X, mean=None, std=None):
    if mean is None or std is None:
        mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, mean, std


def to_tensor(a):
    return torch.tensor(a, dtype=torch.float32)


X, y = load_data("train")
X, mean, std = normalize(X)
X_t, y_t = to_tensor(X), to_tensor(y)

X_test, y_test = load_data("test")
X_test, _, _ = normalize(X_test, mean, std)
X_test_t = to_tensor(X_test)

sizes = [X.shape[1]] + LAYERS
modules = []
for i in range(len(sizes) - 1):
    modules.append(nn.Linear(sizes[i], sizes[i + 1]))
    if i < len(sizes) - 2:
        modules.append(nn.ReLU())
model = nn.Sequential(*modules)

for m in model:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)


criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(WEIGHT))
optimizer = torch.optim.SGD(model.parameters(), lr=LR)  

loss_weighted, loss_plain = [], []
for epoch in range(EPOCHS):
    logits = model(X_t)
    loss = criterion(logits, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_weighted.append(loss.item())
    with torch.no_grad(): 
        loss_plain.append(F.binary_cross_entropy_with_logits(logits, y_t).item())

    if epoch % 500 == 0:
        print(f"epoch {epoch:5d} | weighted loss {loss_weighted[-1]:.4f} | loss non pondérée {loss_plain[-1]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(loss_plain, label="non-weighted loss")
plt.plot(loss_weighted, label=f"weighted loss (weight={WEIGHT:g})", alpha=0.7)
plt.xlabel("iteration")
plt.ylabel("log loss")
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    y_pred = (model(X_test_t) >= 0).numpy().astype(int)  

TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))

accuracy = (TP + TN) / len(y_test)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
rappel = TP / (TP + FN) if (TP + FN) > 0 else 0

print("\n--- CONFUSION MATRIX ---")
print(f"(TP) : {TP}")
print(f"(TN) : {TN}")
print(f"(FP) : {FP}")
print(f"(FN) : {FN}")
print("----------------------------")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {rappel * 100:.2f}%")