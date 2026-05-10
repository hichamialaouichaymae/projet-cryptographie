import pandas as pd
import networkx as nx
import math
import random

# 1. Chargement des données
# On charge les caractéristiques des courbes et les liens du graphe
df_nodes = pd.read_csv("../datasets/node_features.csv")
df_edges = pd.read_csv("../datasets/graph_edges.csv")

# 2. Reconstruction du graphe
G = nx.Graph()
for _, row in df_edges.iterrows():
    G.add_edge(int(row["source_node"]), int(row["target_node"]))

# Transformation en dictionnaire pour un accès rapide aux caractéristiques (j-invariants)
node_dict = df_nodes.set_index('node_id').to_dict('index')
nodes_ids = list(node_dict.keys())

training_data = []
target_size = 100000  # Objectif : 100k exemples pour l'entraînement

print(f"Génération massive du dataset : Objectif {target_size} lignes...")

# 3. Stratégie d'échantillonnage intensif
# On continue de générer des chemins jusqu'à atteindre la taille cible
while len(training_data) < target_size:
    # Sélection de deux nœuds au hasard (Source et Cible)
    source, target = random.sample(nodes_ids, 2)
    
    try:
        # Calcul du chemin optimal (Dijkstra) que l'IA doit apprendre à imiter
        path = nx.shortest_path(G, source=source, target=target)
        
        for i in range(len(path) - 1):
            curr_id = path[i]
            next_id = path[i+1]
            
            curr_node = node_dict[curr_id]
            targ_node = node_dict[target]

            # Calcul de la distance euclidienne entre les j-invariants
            distance = math.sqrt(
                (curr_node["j_real"] - targ_node["j_real"])**2 +
                (curr_node["j_imag"] - targ_node["j_imag"])**2
            )

            # Ajout des données en respectant la structure de ton tableau de référence
            training_data.append({
                "current_node_id": curr_id,
                "target_node_id": target,
                "current_j_real": curr_node["j_real"],
                "current_j_imag": curr_node["j_imag"],
                "target_j_real": targ_node["j_real"],
                "target_j_imag": targ_node["j_imag"],
                "euclidean_j_distance": distance,
                "current_num_neighbors": G.degree(curr_id),       # Ajouté
                "target_num_neighbors": G.degree(target),         # Ajouté
                "candidate_isogeny_degree": 2,                    # Ajouté
                "candidate_isogeny_cost": 1.0,                    # Ajouté
                "shortest_path_length": len(path) - 1 - i,
                "best_next_action": next_id                       # Label à prédire
            })
            
            # Arrêt immédiat si l'objectif est atteint
            if len(training_data) >= target_size: 
                break
                
        if len(training_data) % 10000 == 0:
            print(f"Progression : {len(training_data)}/{target_size}")
            
    except nx.NetworkXNoPath:
        # Ignorer si aucun chemin n'existe entre les deux nœuds
        continue

# 4. Sauvegarde finale
df_final = pd.DataFrame(training_data)
df_final.to_csv("../datasets/final_dataset.csv", index=False)

print(f"✅ Terminé ! {len(df_final)} exemples sauvegardés dans final_dataset.csv.")