import pandas as pd

# 1. Paramètres (p=12007 -> environ 1000 courbes supersingulières possibles)
p = 12007 
F2.<i> = GF(p^2, modulus=x^2+1)

nodes_data = []
curves_found = {} 
queue = []

# 2. Recherche de la courbe "graine" (Seed)
print("Recherche de la courbe de départ...")
for a in range(1, p):
    E = EllipticCurve(F2, [a, 0])
    if E.is_supersingular():
        j = E.j_invariant()
        curves_found[j] = E
        queue.append(E)
        break

# 3. Exploration exhaustive du graphe (BFS)
print("Début de l'exploration du graphe...")
while queue:
    E_curr = queue.pop(0)
    
    try:
        # On utilise les isogénies de degré 2 (standard SIDH/SIKE)
        isogenies = E_curr.isogenies_prime_degree(2)
        
        for phi in isogenies:
            E_next = phi.codomain()
            j_next = E_next.j_invariant()
            
            if j_next not in curves_found:
                # Vérification mathématique de supersingularité
                if E_next.is_supersingular():
                    curves_found[j_next] = E_next
                    queue.append(E_next)
                    
                    if len(curves_found) % 100 == 0:
                        print(f"Courbes trouvées : {len(curves_found)}")
    except:
        continue

# 4. Extraction des caractéristiques pour l'IA (Normalisation)
for idx, (j, E) in enumerate(curves_found.items()):
    coeffs = j.list()
    # j_real et j_imag normalisés entre 0 et 1 pour le modèle de Rayhana
    jr = float(coeffs[0]) / p
    ji = float(coeffs[1]) / p if len(coeffs) > 1 else 0.0
    
    nodes_data.append({
        "node_id": idx,
        "j_real": jr,
        "j_imag": ji
    })

# 5. Sauvegarde dans le dossier datasets
df = pd.DataFrame(nodes_data)
df.to_csv("../datasets/node_features.csv", index=False)

print(f"\n✅ Succès ! {len(nodes_data)} courbes extraites.")
print(f"Fichier sauvegardé : datasets/node_features.csv")