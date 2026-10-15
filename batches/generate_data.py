import json
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

def generate_ball_dataset(n_samples=800, seed=42):
    np.random.seed(seed)
    
    # 1. Génération uniforme des points dans TOUT l'espace 3D
    # (Pas de séparation artificielle d'intervalles)
    diametre = np.random.uniform(20.0, 250.0, n_samples)
    elasticite = np.random.uniform(0.01, 0.95, n_samples)
    
    X = np.column_stack([diametre, elasticite])
    
    # 2. Définition de la sphère rouge au centre de cet espace
    center = np.array([135.0, 0.50])  # Centre exact de l'espace
    
    # Normalisation des coordonnées pour avoir une vraie sphère isotrope (entre -1 et 1)
    X_norm = (X - center) / np.array([115.0, 0.47])
    
    # Distance euclidienne au centre
    distances = np.linalg.norm(X_norm, axis=1)
    
    # 3. Probabilité d'être rouge (1) selon la distance :
    # Rayon = 0.45, avec une zone de flou à la surface
    radius = 0.45
    fuzziness = 0.15  # Plus cette valeur est grande, plus les bords s'entremêlent
    
    # Sigmoïde inverse : 1 au centre, 0 aux bords, flou au rayon
    p_ball = 1.0 / (1.0 + np.exp((distances - radius) / fuzziness))
    
    # Tirage de Bernoulli pour attribuer le label 0 ou 1
    is_ball = np.random.binomial(1, p_ball)
    
    # Formatage des données
    samples = []
    for i in range(n_samples):
        samples.append([
            float(np.round(X[i, 0], 1)),
            float(np.round(X[i, 1], 2)),
            int(is_ball[i])
        ])
        
    return samples

# Génération des fichiers JSON
train_data = generate_ball_dataset(5000, seed=42)
test_data = generate_ball_dataset(5000, seed=100)

train_path = os.path.join(script_dir, "train_data.json")
test_path = os.path.join(script_dir, "test_data.json")

with open(train_path, "w") as f:
    json.dump(train_data, f, indent=2)

with open(test_path, "w") as f:
    json.dump(test_data, f, indent=2)

print("Dataset généré avec succès ! La masse rouge est maintenant au cœur du volume bleu.")