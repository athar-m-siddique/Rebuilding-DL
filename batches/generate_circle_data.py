import numpy as np
import json
import os

def generate_circle_data(n_samples=800, radius=1.0, z_value=1.0):
    # Tirage d'angles θ aléatoires entre 0 et 2π
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    
    # Coordonnées du cercle dans le plan 2D
    x1 = radius * np.cos(theta)
    x2 = radius * np.sin(theta)
    
    # Troisième coordonnée constante à z_value
    x3 = np.full(n_samples, z_value)
    
    # Assemblage sous forme de matrice (800, 3)
    data_matrix = np.column_stack((x1, x2, x3))
    
    # Conversion en listes Python standard pour la sérialisation JSON
    data_list = data_matrix.tolist()
    
    return data_list

if __name__ == "__main__":
    # Génération des 800 points
    points = generate_circle_data(n_samples=800, radius=1.0, z_value=1.0)
    
   # Récupère le dossier absolu du fichier script actuel
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "train_data.json")

    with open(filepath, "w") as f:
        json.dump(points, f, indent=4)

    print(f"✅ {len(points)} points enregistrés dans : {filepath}")