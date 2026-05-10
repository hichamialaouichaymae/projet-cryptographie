🛡️ Isogeny-AI : Cryptographie & Deep LearningCe projet étudie les graphes d'isogénies supersingulières, base de la cryptographie post-quantique, en utilisant l'intelligence artificielle pour analyser leur structure complexe.
👥 Équipe & Rôles
Chaymae HICHAMI-ALAOUI : Responsable de la partie mathématique (SageMath) et de la reconstruction géométrique des courbes.
Rayhana LAZNASNI : Responsable de la préparation des datasets et de la mise en place des modèles de Machine Learning.
📁 Guide des Fichiers
1. ⚙️ Scripts 
(Logique du projet)build_graph.sage : Construit le réseau de courbes et leurs liens (isogénies).
generate_training_dataset.sage : Transforme les données mathématiques en exemples lisibles par une IA.
visualiser_courbe_supersing.py : Permet de voir la forme réelle d'une courbe (y^2 = x^3 + Ax + B) à partir de son invariant j.
2. 📊 Datasets (Données générées)
node_features.csv : Contient l'empreinte digitale (invariant j) de chaque courbe.
graph_edges.csv : Définit "qui est connecté à qui" dans le graphe.
final_dataset.csv : La base de données finale regroupant toutes les informations pour l'entraînement.
🚀 Fonctionnement simplifié
Génération : On crée des milliers de courbes avec generate_curves.sage.
Lien : On établit les connexions cryptographiques via build_graph.sage.
Apprentissage : L'IA utilise final_dataset.csv pour apprendre à naviguer dans ce labyrinthe mathématique et prédir le meilleur chemin .
