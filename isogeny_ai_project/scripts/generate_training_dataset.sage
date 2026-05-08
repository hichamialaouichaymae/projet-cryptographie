import pandas as pd
import networkx as nx
import math
import random

# 1. Chargement des composants du labyrinthe
df_nodes = pd.read_csv("../datasets/node_features.csv")
df_edges = pd.read_csv("../datasets/graph_edges.csv")

# 2. Reconstruction du graphe mathématique
G = nx.Graph()
for _, row in df_edges.iterrows():
    G.add_edge(int(row["source_node"]), int(row["target_node"]))

# Transformation des nœuds en dictionnaire pour un accès ultra-rapide
node_dict = df_nodes.set_index('node_id').to_dict('index')
nodes_ids = list(node_dict.keys())

training_data = []

print("Génération du dataset d'apprentissage en cours...")

# 3. Stratégie de génération par trajets (plus robuste pour l'IA)
# On sélectionne 1000 paires de (départ, cible) au hasard
num_trajectories = 1000 

for _ in range(num_trajectories):
    source, target = random.sample(nodes_ids, 2)
    
    try:
        # Calcul du chemin le plus court (la vérité terrain)
        path = nx.shortest_path(G, source=source, target=target)
        
        # Pour chaque étape du chemin, on enregistre l'état actuel vers la cible
        for i in range(len(path) - 1):
            curr_id = path[i]
            next_id = path[i+1] # L'action parfaite à prédire
            
            curr_node = node_dict[curr_id]
            targ_node = node_dict[target]

            # Calcul de la distance euclidienne (Feature géométrique)
            distance = math.sqrt(
                (curr_node["j_real"] - targ_node["j_real"])**2 +
                (curr_node["j_imag"] - targ_node["j_imag"])**2
            )

            training_data.append({
                "current_node_id": curr_id,
                "target_node_id": target,
                "current_j_real": curr_node["j_real"],
                "current_j_imag": curr_node["j_imag"],
                "target_j_real": targ_node["j_real"],
                "target_j_imag": targ_node["j_imag"],
                "euclidean_j_distance": distance,
                "current_num_neighbors": G.degree(curr_id),
                "target_num_neighbors": G.degree(target),
                "candidate_isogeny_degree": 2, # Degré constant utilisé dans le graphe
                "candidate_isogeny_cost": 1.0, # Coût normalisé
                "shortest_path_length": len(path) - 1 - i, # Nombre d'étapes restantes
                "best_next_action": next_id
            })
            
    except nx.NetworkXNoPath:
        continue

# 4. Sauvegarde finale
df_final = pd.DataFrame(training_data)
df_final.to_csv("../datasets/final_dataset.csv", index=False)

print(f"Succès ! Dataset créé avec {len(df_final)} lignes d'entraînement.")
