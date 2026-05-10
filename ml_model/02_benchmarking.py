import torch
import torch.nn as nn
import numpy as np
import time
import json
import os

# --- ARCHITECTURE (Doit être identique au modèle entraîné) ---
class IsogenyClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(IsogenyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = 'ml_model'
MODEL_PATH = os.path.join(BASE_DIR, 'pytorch_model.pth')
DATA_X_TEST = os.path.join(BASE_DIR, 'X_test.npy')
OUTPUT_METRICS = os.path.join(BASE_DIR, 'benchmark_results.json')

def run_benchmark():
    print("="*80)
    print("ÉTAPE 3 : BENCHMARKING DES PERFORMANCES IA")
    print("="*80)

    # 1. VÉRIFICATION DES FICHIERS
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erreur : Modèle non trouvé à {MODEL_PATH}")
        return
    if not os.path.exists(DATA_X_TEST):
        print(f"❌ Erreur : Fichier de test non trouvé à {DATA_X_TEST}")
        return

    # 2. CHARGEMENT DES DONNÉES
    print(f"\n[1] Chargement des données de test...")
    X_test = np.load(DATA_X_TEST)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 3. CHARGEMENT DU MODÈLE
    print(f"[2] Chargement du modèle PyTorch...")
    input_size = X_test.shape[1]
    
    # --- LIGNE CORRIGÉE ICI ---
    num_classes = 1001  
    # --------------------------

    model = IsogenyClassifier(input_size, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # 4. MESURE VITESSE IA
    print(f"[3] Test de vitesse sur {len(X_test)} prédictions...")
    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    end_time = time.perf_counter()
    
    total_ia_time = end_time - start_time
    avg_ia_time_ms = (total_ia_time / len(X_test)) * 1000

    # 5. SIMULATION VITESSE CLASSIQUE
    CLASSICAL_TIME_PER_OP_MS = 150.0 
    
    # 6. AFFICHAGE DES RÉSULTATS
    print("\n" + " " * 20 + "📊 RÉSULTATS COMPARATIFS")
    print("-" * 60)
    print(f"⏱️  Vitesse IA (moyenne par nœud)      : {avg_ia_time_ms:.6f} ms")
    print(f"⏱️  Vitesse Classique (estimée)        : {CLASSICAL_TIME_PER_OP_MS:.2f} ms")
    print("-" * 60)

    speedup = CLASSICAL_TIME_PER_OP_MS / avg_ia_time_ms
    efficiency = (1 - (avg_ia_time_ms / CLASSICAL_TIME_PER_OP_MS)) * 100

    print(f"🚀 L'IA est {speedup:.1f} fois plus rapide !")
    print(f"⚡ Réduction du temps de calcul : {efficiency:.2f}%")
    print("-" * 60)

    # 7. SAUVEGARDE
    results = {
        "samples": len(X_test),
        "avg_ia_time_ms": avg_ia_time_ms,
        "classical_time_ms": CLASSICAL_TIME_PER_OP_MS,
        "speedup": float(speedup),
        "efficiency_gain_percent": float(efficiency)
    }
    
    with open(OUTPUT_METRICS, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Rapport sauvegardé dans : {OUTPUT_METRICS}")
    print("="*80)

if __name__ == "__main__":
    run_benchmark()