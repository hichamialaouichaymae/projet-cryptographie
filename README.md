# 🔐 IA & Cryptographie Post-Quantique : Optimisation des Isogénies

Ce projet de recherche et développement explore l'intersection entre la **cryptographie avancée** et l'**intelligence artificielle** pour répondre aux défis de la sécurité post-quantique.

## 🎓 Contexte Académique
Projet réalisé par les étudiants ingénieurs de l'**École Nationale des Sciences Appliquées (ENSA)**, au sein du cycle ingénieur en **Cybersécurité et Intelligence Artificielle**.

## 👥 L'Équipe
* **Chaymae Hichami Alaoui (Personne A)** : *Architecte des Données et du Domaine*. 
  * Focus : Mathématiques fondamentales, environnements SageMath, génération de courbes supersingulières et Feature Engineering.
* **Rayhana Laznasni (Personne B)** : *Ingénieur IA et Optimisation*. 
  * Focus : Architecture Deep Learning (PyTorch), Normalisation de données, Benchmarking de performance et optimisation d'inférence.

## 🚀 Performances Records
Notre approche hybride a permis d'atteindre des résultats de performance exceptionnels, prouvant la viabilité de l'IA pour le chiffrement post-quantique :

| Métrique | Algorithme Classique | Modèle IA (Notre Projet) |
| :--- | :--- | :--- |
| **Latence par opération** | 150.00 ms | **0.00404 ms** |
| **Accélération** | 1x | **🚀 37 109x** |
| **Gain d'efficacité** | - | **99.99%** |



## 🛠️ Répartition du Travail

### 🏛️ Architecture & Théorie (Chaymae)
1.  **Moteur de Simulation** : Développement sous **SageMath** pour l'automatisation de la création de paires de courbes $(E_{start}, E_{end})$.
2.  **Stratégie de Labellisation** : Identification des chemins optimaux (isogénies) servant de référence (labels) pour l'apprentissage.
3.  **Feature Engineering** : Traduction des propriétés algébriques des courbes en vecteurs numériques exploitables par l'IA.

### 🤖 Intelligence & Optimisation (Rayhana)
1.  **Pipeline de Données** : Prétraitement, normalisation et gestion du split Training/Testing des données mathématiques.
2.  **Développement du Modèle** : Conception d'un réseau de neurones multicouches (MLP) optimisé pour prédire le prochain saut d'isogénie.
3.  **Benchmarking & Métriques** : Développement des outils de mesure de gain de temps réel et analyse de la valeur ajoutée technologique.

## 🧠 Pourquoi ce projet ?
Le protocole **SIDH (Supersingular Isogeny Diffie-Hellman)** est robuste face à la menace de l'ordinateur quantique, mais il est intrinsèquement lent. Ce projet démontre qu'en tant qu'ingénieurs en **Cybersécurité et IA**, nous pouvons utiliser le Machine Learning pour transformer un protocole théoriquement sûr en une solution pratiquement rapide.

## 📦 Installation et Utilisation
```bash
# 1. Installer les dépendances
pip install torch numpy pandas scikit-learn

# 2. Exécuter le pipeline (Scripts de Rayhana)
python .\ml_model\01_preprocessing.py
python .\ml_model\02_model.py
python .\ml_model\03_benchmarking.py
```
*Ce projet a été réalisé dans le cadre du module de Cryptographie au sein de l'École Nationale des Sciences Appliquées (ENSA) de Béni Mellal.*

© 2026 - Rayhana Laznasni & Chaymae Hichami Alaoui 
