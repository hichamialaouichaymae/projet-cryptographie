import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


print("ÉTAPE 1 : PRÉTRAITEMENT DES DONNÉES")


# 1. CHARGEMENT DES DONNÉES
print("\n[1] Chargement des données...")
df = pd.read_csv('isogeny_ai_project/datasets/final_dataset.csv')
print(f"!! Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# 2. AFFICHAGE DES INFOS
print("\n[2] Informations sur le dataset:")
print(f"   - Nombre d'exemples: {len(df)}")
print(f"   - Nombre de features: {len(df.columns) - 1}")
print(f"   - Cible: 'best_next_action'")
print(f"   - Valeurs manquantes: {df.isnull().sum().sum()}")

# 2. SÉLECTION DES FEATURES ET CIBLE
print("\n[3] Sélection des features...")
feature_columns = [
    'current_j_real', 
    'current_j_imag', 
    'target_j_real', 
    'target_j_imag',
    'euclidean_j_distance', 
    'current_num_neighbors', 
    'target_num_neighbors',
    'candidate_isogeny_degree', 
    'candidate_isogeny_cost', 
    'shortest_path_length'
]

X = df[feature_columns].values  # Features (entrées)
y = df['best_next_action'].values  # Cible (sortie à prédire)

print(f" Features shape: {X.shape}")
print(f" Target shape: {y.shape}")

# 4. NORMALISATION (TRÈS IMPORTANT)
print("\n[4] Normalisation des données...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(" Données normalisées avec StandardScaler")
print(f"   - Mean: {X_scaled.mean(axis=0)[:3]}... (doit être ~0)")
print(f"   - Std: {X_scaled.std(axis=0)[:3]}... (doit être ~1)")

# 5. SPLIT TRAIN/TEST
print("\n[5] Division train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42
)

print(f" Train set: {X_train.shape[0]} exemples")
print(f" Test set: {X_test.shape[0]} exemples")

# 6. SAUVEGARDE DES FICHIERS
print("\n[6] Sauvegarde des données prétraitées...")

# Sauvegarde les données scalées
np.save('ml_model/X_train.npy', X_train)
np.save('ml_model/X_test.npy', X_test)
np.save('ml_model/y_train.npy', y_train)
np.save('ml_model/y_test.npy', y_test)

# Sauvegarde le scaler (pour utiliser plus tard)
with open('ml_model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Sauvegarde les noms des features
with open('ml_model/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("---> Fichiers sauvegardés dans 'ml_model/':")
print("   - X_train.npy")
print("   - X_test.npy")
print("   - y_train.npy")
print("   - y_test.npy")
print("   - scaler.pkl")
print("   - feature_columns.pkl")

# 7. STATISTIQUES FINALES
print("\n[7] Statistiques finales:")
print(f"\n   Target (best_next_action):")
print(f"   - Min: {y.min()}")
print(f"   - Max: {y.max()}")
print(f"   - Mean: {y.mean():.2f}")
print(f"   - Std: {y.std():.2f}")

print(f"\n   Train/Test split:")
print(f"   - Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   - Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 8. VÉRIFICATION DU RATIO TRAIN/TEST
train_ratio = len(X_train) / len(X)
test_ratio = len(X_test) / len(X)
print(f"\n   Ratio: {train_ratio:.1%} train / {test_ratio:.1%} test ✅")

print("\n" + "="*80)
print(" PREPROCESSING TERMINÉ !")
print("="*80)
print("\nFichiers prêts pour l'étape suivante (Modèle).")