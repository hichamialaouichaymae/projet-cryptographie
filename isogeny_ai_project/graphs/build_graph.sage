import pandas as pd
import networkx as nx

# 1. Configuration (Mêmes paramètres que pour tes nœuds)
p = 12007 
F2.<i> = GF(p^2, modulus=x^2+1)

# Charger les nœuds existants pour connaître les j-invariants
df_nodes = pd.read_csv("../datasets/node_features.csv")

# On crée un dictionnaire pour retrouver l'ID à partir du j-invariant
# On multiplie par p car les données étaient normalisées (0-1)
j_to_id = {}
for _, row in df_nodes.iterrows():
    jr = int(round(row["j_real"] * p))
    ji = int(round(row["j_imag"] * p))
    j_val = F2(jr + ji*i)
    j_to_id[j_val] = int(row["node_id"])

G = nx.Graph()
edges_list = []

print("Construction du graphe par isogénies réelles...")

# 2. Parcourir chaque courbe et trouver ses voisines mathématiques
for j_start, id_start in j_to_id.items():
    try:
        # On recrée la courbe à partir de son invariant j
        E = EllipticCurve_from_j(j_start)
        
        # On calcule les isogénies de degré 2 (les routes réelles)
        for phi in E.isogenies_prime_degree(2):
            j_next = phi.codomain().j_invariant()
            
            # Si la voisine est dans notre dataset, on crée le lien
            if j_next in j_to_id:
                id_next = j_to_id[j_next]
                if not G.has_edge(id_start, id_next):
                    G.add_edge(id_start, id_next)
                    edges_list.append({
                        "source_node": id_start, 
                        "target_node": id_next
                    })
    except:
        continue

# 3. Sauvegarde
df_edges = pd.DataFrame(edges_list)
df_edges.to_csv("../datasets/graph_edges.csv", index=False)

print(f"Succès ! {len(edges_list)} arêtes réelles générées.")
