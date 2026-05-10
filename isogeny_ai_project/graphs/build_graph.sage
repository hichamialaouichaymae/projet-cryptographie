import pandas as pd
import random

# 1. Configuration
p = 12007 
F2.<i> = GF(p^2, modulus=x^2+1)

# Charger les nœuds
df_nodes = pd.read_csv("../datasets/node_features.csv")
j_to_id = {}
id_to_curve = {} # On stocke les courbes pour aller vite

print("Chargement des courbes en mémoire...")
for _, row in df_nodes.iterrows():
    jr = int(round(row["j_real"] * p))
    ji = int(round(row["j_imag"] * p))
    j_val = F2(jr + ji*i)
    id_start = int(row["node_id"])
    j_to_id[j_val] = id_start
    id_to_curve[id_start] = EllipticCurve_from_j(j_val)

edges_list = []
target_count = 100000 # Ton objectif

print(f"Génération de {target_count} exemples de chemins (Random Walks)...")

node_ids = list(id_to_curve.keys())

while len(edges_list) < target_count:
    # On choisit un point de départ au hasard
    curr_id = random.choice(node_ids)
    E = id_to_curve[curr_id]
    
    try:
        # On calcule les voisines (Degré 2 pour SIDH/SIKE)
        isogenies = E.isogenies_prime_degree(2)
        if not isogenies: continue
        
        # On simule un saut
        phi = random.choice(isogenies)
        j_next = phi.codomain().j_invariant()
        
        if j_next in j_to_id:
            next_id = j_to_id[j_next]
            
            edges_list.append({
                "source_node": curr_id, 
                "target_node": next_id,
                "label": 1 # Indique un chemin valide/optimal
            })
            
            if len(edges_list) % 10000 == 0:
                print(f"Progression : {len(edges_list)}/{target_count}")
    except:
        continue

# 3. Sauvegarde
df_edges = pd.DataFrame(edges_list)
df_edges.to_csv("../datasets/graph_edges.csv", index=False)

print(f"Succès ! Dataset de {len(edges_list)} arêtes prêt pour l'IA.")