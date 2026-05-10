<<<<<<< HEAD
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Charger les données
df_edges = pd.read_csv("../datasets/graph_edges.csv")
G = nx.from_pandas_edgelist(df_edges, source='source_node', target='target_node')

print(f"Visualisation d'un sous-ensemble du graphe (les 200 premières arêtes)...")

plt.figure(figsize=(12, 12))
# On prend un sous-ensemble pour que ce soit joli, ou G entier si ton PC est puissant
sub_G = G.edge_subgraph(list(G.edges())[:500]) 

pos = nx.spring_layout(sub_G, k=0.15, iterations=20)
nx.draw(sub_G, pos, node_size=20, node_color='red', edge_color='gray', alpha=0.6, with_labels=False)

plt.title("Aperçu du Graphe d'Isogénies (Labyrinthe Cryptographique)")
=======
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Charger les données
df_edges = pd.read_csv("../datasets/graph_edges.csv")
G = nx.from_pandas_edgelist(df_edges, source='source_node', target='target_node')

print(f"Visualisation d'un sous-ensemble du graphe (les 200 premières arêtes)...")

plt.figure(figsize=(12, 12))
# On prend un sous-ensemble pour que ce soit joli, ou G entier si ton PC est puissant
sub_G = G.edge_subgraph(list(G.edges())[:500]) 

pos = nx.spring_layout(sub_G, k=0.15, iterations=20)
nx.draw(sub_G, pos, node_size=20, node_color='red', edge_color='gray', alpha=0.6, with_labels=False)

plt.title("Aperçu du Graphe d'Isogénies (Labyrinthe Cryptographique)")
>>>>>>> 9e4e29d9a8cc8f66b2d5ab41b3f92813c87e6d8a
plt.show()