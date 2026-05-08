import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger ton dataset
df_nodes = pd.read_csv("datasets/node_features.csv")

# 2. Choisir une courbe (par exemple la première du fichier)
target_id = 0 
node_data = df_nodes[df_nodes['node_id'] == target_id].iloc[0]

# 3. Retrouver l'invariant j (on dé-normalise si nécessaire)
# Ici on utilise une valeur illustrative pour le dessin continu
j = node_data['j_real'] 

# Formule simplifiée pour retrouver A et B à partir de j
# Dans le monde réel (complexe), on utilise souvent y² = x³ + [3j/(1728-j)]x + [2j/(1728-j)]
k = j / (1728 - j) if j != 1728 else 1
A = 3 * k
B = 2 * k

# 4. Générer les points de la courbe pour le dessin
x = np.linspace(-10, 10, 2000)
y2 = x**3 + A*x + B


# On ne garde que les zones où y² est positif
x_valid = x[y2 >= 0]
y_vals = np.sqrt(y2[y2 >= 0])

# 5. Visualisation
plt.figure(figsize=(10, 7))
plt.plot(x_valid, y_vals, color='#2ecc71', linewidth=2, label=f"Courbe ID {target_id}")
plt.plot(x_valid, -y_vals, color='#2ecc71', linewidth=2) # Symétrie
# 2. On ajuste les limites du graphique pour tout voir
plt.xlim(-10, 10)
plt.ylim(-20, 20)

plt.title(f"Reconstruction de la Courbe Elliptique (ID: {target_id})", fontsize=14)
plt.axhline(0, color='black', alpha=0.3)
plt.axvline(0, color='black', alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()