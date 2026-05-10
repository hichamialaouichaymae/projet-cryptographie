import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import time


print("="*80)
print("ÉTAPE 2 : ENTRAÎNEMENT DU MODÈLE IA (PyTorch) - CLASSIFICATION")
print("="*80)


# [1] CHARGEMENT DES DONNÉES
print("\n[1] Chargement des données prétraitées...")
X_train = np.load('ml_model/X_train.npy')
X_test = np.load('ml_model/X_test.npy')
y_train = np.load('ml_model/y_train.npy')
y_test = np.load('ml_model/y_test.npy')

print(f"   - X_train shape: {X_train.shape}")
print(f"   - X_test shape: {X_test.shape}")
print(f"   - y_train shape: {y_train.shape}")
print(f"   - y_test shape: {y_test.shape}")

# Convertir en tenseurs PyTorch
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)  # LongTensor pour CrossEntropyLoss
y_test_tensor = torch.LongTensor(y_test)

print("   ✅ Tenseurs créés")


# [2] CRÉER LES DATALOADERS
print("\n[2] Création des DataLoaders...")
batch_size = 64

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"   - Batch size: {batch_size}")
print(f"   - Train batches: {len(train_loader)}")
print(f"   - Test batches: {len(test_loader)}")


# [3] DÉFINIR LE MODÈLE (CLASSIFICATION)
print("\n[3] Construction du modèle...")

class IsogenyClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(IsogenyClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        
        # Output: probabilités pour chaque classe (1000 nœuds)
        self.fc4 = nn.Linear(32, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.relu(self.fc3(x))
        
        # Pas d'activation ici, CrossEntropyLoss l'ajoute automatiquement
        x = self.fc4(x)
        return x


device = torch.device('cpu')
model = IsogenyClassifier(num_classes=1000).to(device)

print("   ✅ Modèle créé")
print(f"   - Device: {device}")
print(model)


# [4] CONFIGURATION DE L'ENTRAÎNEMENT
print("\n[4] Configuration de l'entraînement...")

num_epochs = 100
learning_rate = 0.001

# Loss function pour CLASSIFICATION
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"   - Epochs: {num_epochs}")
print(f"   - Learning rate: {learning_rate}")
print(f"   - Loss function: CrossEntropyLoss")
print(f"   - Optimizer: Adam")


# [5] ENTRAÎNEMENT
print("\n[5] Entraînement du modèle...")
print("-" * 80)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

start_time = time.time()

for epoch in range(num_epochs):
    # PHASE D'ENTRAÎNEMENT
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiques
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    
    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # PHASE DE TEST
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # Affichage tous les 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

elapsed_time = time.time() - start_time
print("-" * 80)
print(f"✅ Entraînement terminé ! ({elapsed_time:.2f}s)")


# [6] SAUVEGARDE DU MODÈLE
print("\n[6] Sauvegarde du modèle...")
torch.save(model.state_dict(), 'ml_model/pytorch_model.pth')
print("   ✅ Modèle sauvegardé : ml_model/pytorch_model.pth")


# [7] ÉVALUATION FINALE
print("\n[7] Évaluation finale...")

# Évaluation sur l'ensemble du dataset
model.eval()
with torch.no_grad():
    train_outputs = model(X_train_tensor.to(device))
    test_outputs = model(X_test_tensor.to(device))
    
    _, train_preds = torch.max(train_outputs, 1)
    _, test_preds = torch.max(test_outputs, 1)
    
    train_accuracy = (train_preds == y_train_tensor.to(device)).float().mean().item()
    test_accuracy = (test_preds == y_test_tensor.to(device)).float().mean().item()
    
    # MAE (Mean Absolute Error) - à quel point on se trompe en moyenne
    train_mae = torch.abs(train_preds.float() - y_train_tensor.float().to(device)).mean().item()
    test_mae = torch.abs(test_preds.float() - y_test_tensor.float().to(device)).mean().item()

print(f"   Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"   Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Train MAE: {train_mae:.4f}")
print(f"   Test MAE:  {test_mae:.4f}")


# [8] SAUVEGARDE DES MÉTRIQUES
print("\n[8] Sauvegarde des métriques...")

metrics = {
    "model_type": "PyTorch Classification (CrossEntropyLoss)",
    "architecture": "IsogenyClassifier",
    "input_features": 10,
    "output_classes": 1000,
    "hidden_layers": [128, 64, 32],
    "dropout": 0.2,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss",
    "training_time_seconds": elapsed_time,
    "final_metrics": {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "final_train_loss": float(train_losses[-1]),
        "final_test_loss": float(test_losses[-1])
    },
    "history": {
        "train_losses": [float(x) for x in train_losses],
        "test_losses": [float(x) for x in test_losses],
        "train_accuracies": [float(x) for x in train_accuracies],
        "test_accuracies": [float(x) for x in test_accuracies]
    }
}

with open('ml_model/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("   ✅ Métriques sauvegardées : ml_model/model_metrics.json")


# [9] RÉSUMÉ FINAL
print("\n" + "="*80)
print("✅ ENTRAÎNEMENT TERMINÉ !")
print("="*80)

print(f"\nRésumé:")
print(f"  - Modèle: ml_model/pytorch_model.pth")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Test MAE: {test_mae:.4f} nœuds")
print(f"  - Temps d'entraînement: {elapsed_time:.2f}s")
print(f"  - Prochaine étape: Benchmarking (02_benchmarking.py)")

print("\n" + "="*80)
