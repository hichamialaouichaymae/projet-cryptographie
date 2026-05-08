import pandas as pd
import matplotlib.pyplot as plt

df_nodes = pd.read_csv("../datasets/node_features.csv")

plt.figure(figsize=(10, 7))
plt.scatter(df_nodes['j_real'], df_nodes['j_imag'], alpha=0.6, s=10, c='blue')
plt.title("Répartition des courbes supersingulières (Plan des invariants j)")
plt.xlabel("j_real (Normalisé)")
plt.ylabel("j_imag (Normalisé)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()