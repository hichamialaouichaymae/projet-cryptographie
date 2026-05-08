import pandas as pd

# 1. Paramètres
p = 12007 # Nombre premier (p = 3 mod 4 pour faciliter la supersingularité)
F2.<i> = GF(p^2, modulus=x^2+1)

# Liste pour stocker nos résultats
nodes_data = []
curves_found = {} # Pour éviter les doublons via j-invariant
queue = []

# 2. Trouver la TOUTE PREMIÈRE courbe (La graine)
# On cherche une courbe supersingulière de départ par balayage
print("Recherche de la courbe de départ...")
for a in range(1, p):
    E = EllipticCurve(F2, [a, 0])
    if E.is_supersingular():
        j = E.j_invariant()
        curves_found[j] = E
        queue.append(E)
        break

# 3. Exploration avec vérification mathématique à chaque pas

while len(curves_found) < 1000 and queue:
    E_curr = queue.pop(0)
    
    # On calcule les voisines (isogénies de degré 2)
    try:
        isogenies = E_curr.isogenies_prime_degree(2)
        
        for phi in isogenies:
            E_next = phi.codomain()
            j_next = E_next.j_invariant()
            
            # CONDITION DOUBLE : Nouveau j-invariant ET certification mathématique
            if j_next not in curves_found:
                if E_next.is_supersingular(): # <--- LA VÉRIFICATION QUE TU VOULAIS
                    curves_found[j_next] = E_next
                    queue.append(E_next)
                    
                    if len(curves_found) % 100 == 0:
                        print(f"nombre de courbes touvées : {len(curves_found)}/1000")
            
            if len(curves_found) >= 1000:
                break
    except:
        continue

# 4. Transformation en Dataset pour l'IA
for idx, (j, E) in enumerate(curves_found.items()):
    coeffs = j.list()
    # On normalise pour ton partenaire (valeurs entre 0 et 1)
    jr = float(coeffs[0]) / p
    ji = float(coeffs[1]) / p if len(coeffs) > 1 else 0.0
    
    nodes_data.append({
        "node_id": idx,
        "j_real": jr,
        "j_imag": ji
    })

# 5. Sauvegarde
df = pd.DataFrame(nodes_data)
df.to_csv("../datasets/node_features.csv", index=False)
print(f"Succès ! Dataset créé avec {len(nodes_data)} courbes supersingulières.")
