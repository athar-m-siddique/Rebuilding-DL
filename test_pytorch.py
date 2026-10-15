import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATA_DIR = "batches"   # même dossier que main_2.py
LR = 0.1
WEIGHT = 1          # poids appliqué aux exemples positifs (y == 1)
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


# ---------- Données ----------
X, y = load_data("train")
X, mean, std = normalize(X)
X_t, y_t = to_tensor(X), to_tensor(y)

X_test, y_test = load_data("test")
X_test, _, _ = normalize(X_test, mean, std)
X_test_t = to_tensor(X_test)

# ---------- Réseau : Dense/ReLU x3 + Dense (la sigmoïde est dans la loss) ----------
sizes = [X.shape[1]] + LAYERS
modules = []
for i in range(len(sizes) - 1):
    modules.append(nn.Linear(sizes[i], sizes[i + 1]))
    if i < len(sizes) - 2:
        modules.append(nn.ReLU())
model = nn.Sequential(*modules)

# Initialisation He (std = sqrt(2 / n_entrées)), biais à 0, comme main_2.py
for m in model:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

# BCE avec sigmoïde intégrée, pondérée sur les positifs.
# Gradient par rapport au logit : (p - y)/m pour y=0 et weight*(p - y)/m pour y=1, comme main_2.py
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(WEIGHT))
optimizer = torch.optim.SGD(model.parameters(), lr=LR)   # descente de gradient batch complet

# ---------- Entraînement ----------
loss_weighted, loss_plain = [], []
for epoch in range(EPOCHS):
    logits = model(X_t)
    loss = criterion(logits, y_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_weighted.append(loss.item())
    with torch.no_grad():  # loss non pondérée : comparable à celle de main_2.py
        loss_plain.append(F.binary_cross_entropy_with_logits(logits, y_t).item())

    if epoch % 500 == 0:
        print(f"epoch {epoch:5d} | loss pondérée {loss_weighted[-1]:.4f} | loss non pondérée {loss_plain[-1]:.4f}")

# ---------- Courbe de loss ----------
plt.figure(figsize=(8, 5))
plt.plot(loss_plain, label="loss non pondérée (comparable à main_2)")
plt.plot(loss_weighted, label=f"loss pondérée (weight={WEIGHT:g})", alpha=0.7)
plt.xlabel("itération")
plt.ylabel("log loss")
plt.legend()
plt.show()

# ---------- Évaluation sur le jeu de test ----------
model.eval()
with torch.no_grad():
    y_pred = (model(X_test_t) >= 0).numpy().astype(int)   # logit >= 0  <=>  sigmoïde >= 0.5

TP = np.sum((y_pred == 1) & (y_test == 1))
TN = np.sum((y_pred == 0) & (y_test == 0))
FP = np.sum((y_pred == 1) & (y_test == 0))
FN = np.sum((y_pred == 0) & (y_test == 1))

accuracy = (TP + TN) / len(y_test)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
rappel = TP / (TP + FN) if (TP + FN) > 0 else 0

print("\n--- MATRICE DE CONFUSION ---")
print(f"Vrais Positifs  (TP) : {TP}")
print(f"Vrais Négatifs  (TN) : {TN}")
print(f"Faux Positifs   (FP) : {FP}")
print(f"Faux Négatifs   (FN) : {FN}")
print("----------------------------")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {rappel * 100:.2f}%")